<#
.SYNOPSIS
Validate that docs\latex, the local Overleaf mirror, and the pushed Overleaf
remote are in strict sync.

.DESCRIPTION
The source tree is docs\latex. The mirror tree is the Overleaf-connected Git
checkout whose root should contain the projected source files directly. This
test compares exact root entries, exact relative file paths, SHA-256 file
content hashes, referenced manuscript figures, and optionally the pushed remote
Git tree.
#>

[CmdletBinding()]
param(
    [string]$SourceRoot = '',
    [string]$MirrorRoot = 'C:\Users\Tanner\Documents\git\LaTeX-Projects\MEA-Absorption-Column-LaTeX',
    [string[]]$ExcludedSourceEntries = @('scripts', 'builds'),
    [switch]$RequireCleanMirrorGit,
    [switch]$VerifyRemote,
    [string]$RemoteName = 'origin',
    [string]$RemoteBranch = 'master'
)

$ErrorActionPreference = 'Stop'

if ([string]::IsNullOrWhiteSpace($SourceRoot)) {
    $SourceRoot = Join-Path $PSScriptRoot '..'
}

function Resolve-RequiredPath {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][string]$Label
    )

    if (-not (Test-Path -LiteralPath $Path)) {
        throw "$Label does not exist: $Path"
    }

    return (Resolve-Path -LiteralPath $Path).Path
}

function Get-PortableRelativePath {
    param(
        [Parameter(Mandatory = $true)][string]$BasePath,
        [Parameter(Mandatory = $true)][string]$TargetPath
    )

    $baseFullPath = [System.IO.Path]::GetFullPath($BasePath)
    $targetFullPath = [System.IO.Path]::GetFullPath($TargetPath)
    if (-not $baseFullPath.EndsWith([System.IO.Path]::DirectorySeparatorChar)) {
        $baseFullPath += [System.IO.Path]::DirectorySeparatorChar
    }

    $baseUri = [System.Uri]::new($baseFullPath)
    $targetUri = [System.Uri]::new($targetFullPath)
    return [System.Uri]::UnescapeDataString($baseUri.MakeRelativeUri($targetUri).ToString()).Replace('/', [System.IO.Path]::DirectorySeparatorChar)
}

function Get-RootNameSet {
    param(
        [Parameter(Mandatory = $true)][string]$RootPath,
        [string[]]$ExcludeNames = @()
    )

    return @(Get-ChildItem -LiteralPath $RootPath -Force |
        Where-Object { $ExcludeNames -cnotcontains $_.Name } |
        ForEach-Object { $_.Name })
}

function Assert-ExactSet {
    param(
        [Parameter(Mandatory = $true)][string[]]$Expected,
        [Parameter(Mandatory = $true)][string[]]$Actual,
        [Parameter(Mandatory = $true)][string]$Label
    )

    $missing = @($Expected | Where-Object { $Actual -cnotcontains $_ })
    $extra = @($Actual | Where-Object { $Expected -cnotcontains $_ })
    if ($missing.Count -gt 0 -or $extra.Count -gt 0) {
        $messages = New-Object System.Collections.Generic.List[string]
        if ($missing.Count -gt 0) {
            $messages.Add("missing: $($missing -join ', ')")
        }
        if ($extra.Count -gt 0) {
            $messages.Add("extra: $($extra -join ', ')")
        }
        throw "$Label mismatch; $($messages -join '; ')"
    }
}

function Get-ProjectedFileMap {
    param(
        [Parameter(Mandatory = $true)][string]$RootPath,
        [string[]]$ExcludeRootEntries = @()
    )

    $map = @{}
    $rootFullPath = [System.IO.Path]::GetFullPath($RootPath)
    $files = Get-ChildItem -LiteralPath $rootFullPath -Recurse -File -Force
    foreach ($file in $files) {
        $relativePath = Get-PortableRelativePath -BasePath $rootFullPath -TargetPath $file.FullName
        $firstPart = ($relativePath -split '[\\/]')[0]
        if ($ExcludeRootEntries -ccontains $firstPart) {
            continue
        }

        $map[$relativePath] = (Get-FileHash -LiteralPath $file.FullName -Algorithm SHA256).Hash
    }

    return $map
}

function Assert-FileMapMatch {
    param(
        [Parameter(Mandatory = $true)]$ExpectedMap,
        [Parameter(Mandatory = $true)]$ActualMap,
        [Parameter(Mandatory = $true)][string]$Label
    )

    $expectedPaths = @($ExpectedMap.Keys)
    $actualPaths = @($ActualMap.Keys)
    $missing = @($expectedPaths | Where-Object { $actualPaths -cnotcontains $_ })
    $extra = @($actualPaths | Where-Object { $expectedPaths -cnotcontains $_ })
    $changed = @($expectedPaths | Where-Object {
        $actualPaths -ccontains $_ -and $ExpectedMap[$_] -cne $ActualMap[$_]
    })

    if ($missing.Count -gt 0 -or $extra.Count -gt 0 -or $changed.Count -gt 0) {
        $messages = New-Object System.Collections.Generic.List[string]
        if ($missing.Count -gt 0) {
            $messages.Add("missing paths: $($missing -join ', ')")
        }
        if ($extra.Count -gt 0) {
            $messages.Add("extra paths: $($extra -join ', ')")
        }
        if ($changed.Count -gt 0) {
            $messages.Add("hash mismatches: $($changed -join ', ')")
        }
        throw "$Label mismatch; $($messages -join '; ')"
    }
}

function Get-FigureReferences {
    param(
        [Parameter(Mandatory = $true)][string]$RootPath,
        [string[]]$ExcludeRootEntries = @()
    )

    $references = New-Object System.Collections.Generic.List[string]
    $texFiles = Get-ChildItem -LiteralPath $RootPath -Recurse -Filter '*.tex' -File -Force
    foreach ($texFile in $texFiles) {
        $relativePath = Get-PortableRelativePath -BasePath $RootPath -TargetPath $texFile.FullName
        $firstPart = ($relativePath -split '[\\/]')[0]
        if ($ExcludeRootEntries -ccontains $firstPart) {
            continue
        }

        $content = Get-Content -LiteralPath $texFile.FullName -Raw
        $matches = [regex]::Matches($content, '\\includegraphics(?:\[[^\]]*\])?\{(?<path>figures/[^}]+)\}')
        foreach ($match in $matches) {
            $references.Add($match.Groups['path'].Value)
        }
    }

    return @($references | Sort-Object -Unique)
}

function Assert-FigureReferencesExist {
    param(
        [Parameter(Mandatory = $true)][string]$RootPath,
        [Parameter(Mandatory = $true)][string[]]$References,
        [Parameter(Mandatory = $true)][string]$Label
    )

    $missing = New-Object System.Collections.Generic.List[string]
    foreach ($reference in $References) {
        $relativePath = $reference.Replace('/', [System.IO.Path]::DirectorySeparatorChar)
        $expectedPath = Join-Path $RootPath $relativePath
        if (-not (Test-Path -LiteralPath $expectedPath)) {
            $missing.Add($reference)
        }
    }

    if ($missing.Count -gt 0) {
        throw "$Label missing referenced figures: $($missing -join ', ')"
    }
}

function Invoke-Git {
    param(
        [Parameter(Mandatory = $true)][string]$RepositoryPath,
        [Parameter(Mandatory = $true)][string[]]$Arguments
    )

    $output = & git -C $RepositoryPath @Arguments 2>&1
    if ($LASTEXITCODE -ne 0) {
        throw "git $($Arguments -join ' ') failed in ${RepositoryPath}: $($output | Out-String)"
    }

    return @($output)
}

$sourceRootPath = Resolve-RequiredPath -Path $SourceRoot -Label 'LaTeX source root'
$mirrorRootPath = Resolve-RequiredPath -Path $MirrorRoot -Label 'Overleaf mirror root'

if (-not (Test-Path -LiteralPath (Join-Path $mirrorRootPath '.git'))) {
    throw "Mirror root is not a Git checkout: $mirrorRootPath"
}

$sourceRootNames = Get-RootNameSet -RootPath $sourceRootPath -ExcludeNames $ExcludedSourceEntries
$mirrorRootNames = Get-RootNameSet -RootPath $mirrorRootPath -ExcludeNames @('.git')
Assert-ExactSet -Expected $sourceRootNames -Actual $mirrorRootNames -Label 'Mirror root entries'

$sourceFileMap = Get-ProjectedFileMap -RootPath $sourceRootPath -ExcludeRootEntries $ExcludedSourceEntries
$mirrorFileMap = Get-ProjectedFileMap -RootPath $mirrorRootPath -ExcludeRootEntries @('.git')
Assert-FileMapMatch -ExpectedMap $sourceFileMap -ActualMap $mirrorFileMap -Label 'Mirror file projection'

$sourceFigureReferences = Get-FigureReferences -RootPath $sourceRootPath -ExcludeRootEntries $ExcludedSourceEntries
Assert-FigureReferencesExist -RootPath $sourceRootPath -References $sourceFigureReferences -Label 'Source'
Assert-FigureReferencesExist -RootPath $mirrorRootPath -References $sourceFigureReferences -Label 'Mirror'

if ($RequireCleanMirrorGit) {
    $status = Invoke-Git -RepositoryPath $mirrorRootPath -Arguments @('status', '--porcelain')
    if ($status.Count -gt 0) {
        throw "Mirror Git checkout is not clean: $($status -join '; ')"
    }
}

if ($VerifyRemote) {
    Invoke-Git -RepositoryPath $mirrorRootPath -Arguments @('fetch', $RemoteName) | Out-Null
    $remoteRef = "$RemoteName/$RemoteBranch"
    $head = ((Invoke-Git -RepositoryPath $mirrorRootPath -Arguments @('rev-parse', 'HEAD') | Select-Object -First 1).ToString()).Trim()
    $remoteHead = ((Invoke-Git -RepositoryPath $mirrorRootPath -Arguments @('rev-parse', $remoteRef) | Select-Object -First 1).ToString()).Trim()
    if ($head -cne $remoteHead) {
        throw "Mirror HEAD does not match ${remoteRef}: HEAD=$head, ${remoteRef}=$remoteHead"
    }

    $remoteRootNames = @(Invoke-Git -RepositoryPath $mirrorRootPath -Arguments @('ls-tree', '--name-only', $remoteRef))
    Assert-ExactSet -Expected $sourceRootNames -Actual $remoteRootNames -Label "Remote root entries ($remoteRef)"

    $remoteFilePaths = @(Invoke-Git -RepositoryPath $mirrorRootPath -Arguments @('ls-tree', '-r', '--name-only', $remoteRef) |
        ForEach-Object { $_.Replace('/', [System.IO.Path]::DirectorySeparatorChar) })
    $sourceFilePaths = @($sourceFileMap.Keys)
    Assert-ExactSet -Expected $sourceFilePaths -Actual $remoteFilePaths -Label "Remote file paths ($remoteRef)"
}

Write-Host 'Overleaf sync audit passed.'
