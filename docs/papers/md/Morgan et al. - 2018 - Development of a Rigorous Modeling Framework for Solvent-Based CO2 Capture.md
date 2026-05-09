\title{
Development of a Rigorous Modeling Framework for Solvent-Based $\mathrm{CO}_{2}$ Capture. Part 2: Steady-State Validation and Uncertainty Quantification with Pilot Plant Data
}

\author{
Joshua C. Morgan ${ }^{\text {a,c }}$, Anderson Soares Chinen ${ }^{\text {a }}$, Benjamin Omell ${ }^{\text {c }}$, Debangsu Bhattacharyya ${ }^{\text {a }}$, Charles Tong ${ }^{\text {b }}$, David C. Miller ${ }^{\text {c }}$, Bill Buschle ${ }^{\text {d }}$, Mathieu Lucquiaud ${ }^{\text {d }}$ \\ ${ }^{\mathrm{a}}$ Department of Chemical Engineering, West Virginia University, Morgantown, WV 26506, USA \\ ${ }^{\mathrm{b}}$ Lawrence Livermore National Laboratory, Livermore, CA 94550, USA \\ ${ }^{\mathrm{c}}$ National Energy Technology Laboratory, 626 Cochrans Mill Rd, Pittsburgh, PA 15236, USA \\ ${ }^{\mathrm{d}}$ School of Engineering, University of Edinburgh, Edinburgh, EH9 3JL, UK
}

\section*{Supporting Information for Publication}

\begin{table}
\captionsetup{labelformat=empty}
\caption{Table S1. Summary of NCCC steady state data (absorber variables)}
\begin{tabular}{|l|l|l|l|l|l|l|l|l|l|}
\hline Case No. & Lean Solvent Flowrate (kg/hr) & Lean Solvent Temperature $\left({ }^{\circ} \mathrm{C}\right)$ & Lean Solvent $\mathrm{CO}_{2}$ Loading (mol CO2/MEA) & Nominal Lean Solvent MEA Weight Fraction & Gas Flowrate (kg/hr) & Gas Temperature $\left({ }^{\circ} \mathrm{C}\right)$ & $\mathrm{CO}_{2}$ mol\% & Column Pressure (kPa) & Number of Beds (Intercoolers) \\
\hline K1 & 6804 & 40.97 & 0.145 & 0.298 & 2248 & 42.48 & 11.55 & 108.82 & 3 (2) \\
\hline K2 & 11794 & 40.52 & 0.247 & 0.312 & 2243 & 44.94 & 11.4 & 107.06 & 3 (2) \\
\hline K3 & 3175 & 45.68 & 0.091 & 0.295 & 2242 & 43.67 & 10.58 & 107.78 & 3 (2) \\
\hline K4 & 3175 & 46.72 & 0.083 & 0.310 & 2243 & 44.73 & 11.45 & 107.65 & 3 (2) \\
\hline K5 & 6804 & 41.57 & 0.108 & 0.306 & 2235 & 43.78 & 9.19 & 106.94 & 3 (2) \\
\hline K6 & 6804 & 40.87 & 0.347 & 0.307 & 2237 & 42.18 & 9.12 & 107.10 & 3 (2) \\
\hline K7 & 11791 & 40.62 & 0.399 & 0.288 & 2233 & 44.72 & 9.18 & 107.30 & 3 (2) \\
\hline K8 & 11643 & 40.57 & 0.154 & 0.285 & 2237 & 42.47 & 9.2 & 107.26 & 3 (2) \\
\hline K9 & 3175 & 42.66 & 0.239 & 0.311 & 2232 & 44.87 & 9.22 & 107.49 & 3 (2) \\
\hline K10 & 3175 & 48.59 & 0.062 & 0.297 & 2237 & 42.15 & 9.18 & 107.71 & 3 (2) \\
\hline K11 & 6804 & 40.83 & 0.161 & 0.293 & 2237 & 44.40 & 10.01 & 107.20 & 3 (2) \\
\hline K12 & 6804 & 40.92 & 0.160 & 0.293 & 2231 & 43.24 & 8.24 & 106.37 & 3 (2) \\
\hline K13 & 6804 & 41.93 & 0.164 & 0.303 & 2238 & 42.09 & 9.35 & 107.56 & 3 (0) \\
\hline K14 & 8845 & 40.41 & 0.224 & 0.303 & 2895 & 47.67 & 9.01 & 107.86 & 3 (2) \\
\hline K15 & 8845 & 40.60 & 0.224 & 0.313 & 2908 & 42.83 & 9.07 & 108.08 & 3 (2) \\
\hline K16 & 4128 & 44.45 & 0.124 & 0.329 & 2903 & 45.42 & 9.21 & 108.70 & 3 (2) \\
\hline K17 & 6804 & 42.03 & 0.168 & 0.307 & 2240 & 41.03 & 9.19 & 108.12 & 2 (0) \\
\hline K18 & 6804 & 42.24 & 0.141 & 0.302 & 2271 & 46.07 & 10.19 & 109.18 & 1 (0) \\
\hline K19 & 11793 & 40.90 & 0.184 & 0.278 & 1440 & 46.18 & 11 & 108.02 & 1 (0) \\
\hline K20 & 3175 & 45.30 & 0.075 & 0.276 & 1324 & 46.09 & 10.98 & 108.12 & 1 (0) \\
\hline K21 & 3175 & 45.18 & 0.074 & 0.271 & 1366 & 46.11 & 10.18 & 107.76 & 2 (0) \\
\hline K22 & 6804 & 41.24 & 0.130 & 0.281 & 1379 & 46.06 & 10.92 & 107.51 & 2 (1) \\
\hline K23 & 6804 & 41.23 & 0.138 & 0.281 & 1418 & 46.11 & 10.13 & 107.37 & 2 (1) \\
\hline
\end{tabular}
\end{table}

\begin{table}
\captionsetup{labelformat=empty}
\caption{Table S2. Summary of NCCC steady state data (stripper variables)}
\begin{tabular}{|l|l|l|l|l|l|l|l|}
\hline Case No. & Rich Solvent Flowrate (kg/hr) & Inlet Solvent Temperature ( ${ }^{\circ} \mathrm{C}$ ) & Outlet Solvent Temperature ( ${ }^{\circ} \mathrm{C}$ ) & Rich Loading (mol $\mathrm{CO}_{2} / \mathrm{MEA}$ ) & Rich MEA Weight fraction & Operating Pressure (kPa) & Reboiler Duty (kW) \\
\hline K1 & 7242 & 104.8 & 120.18 & 0.384 & 0.300 & 183.87 & 431 \\
\hline K2 & 12284 & 104.8 & 117.43 & 0.385 & 0.314 & 182.06 & 430 \\
\hline K3 & 3335 & 97.4 & 122.15 & 0.475 & 0.313 & 184.36 & 427 \\
\hline K4 & 3343 & 97.6 & 122.53 & 0.470 & 0.328 & 184.15 & 427 \\
\hline K5 & 7212 & 109.0 & 121.68 & 0.295 & 0.308 & 183.43 & 673 \\
\hline K6 & 7063 & 95.7 & 110.21 & 0.469 & 0.309 & 179.88 & 173 \\
\hline K7 & 12092 & 93.2 & 103.76 & 0.471 & 0.289 & 184.41 & 170 \\
\hline K8 & 12043 & 110.1 & 120.35 & 0.275 & 0.289 & 183.45 & 677 \\
\hline K9 & 3337 & 98.4 & 117.69 & 0.474 & 0.318 & 182.84 & 166 \\
\hline K10 & 3358 & 94.5 & 124.00 & 0.477 & 0.309 & 182.67 & 671 \\
\hline K11 & 7241 & 107.4 & 119.88 & 0.378 & 0.294 & 182.67 & 425 \\
\hline K12 & 7173 & 109.2 & 120.56 & 0.341 & 0.294 & 183.81 & 422 \\
\hline K13 & 7078 & 107.6 & 120.65 & 0.360 & 0.307 & 183.19 & 419 \\
\hline K14 & 9393 & 104.8 & 119.35 & 0.420 & 0.302 & 184.02 & 420 \\
\hline K15 & 9349 & 103.5 & 119.32 & 0.413 & 0.313 & 182.00 & 420 \\
\hline K16 & 4347 & 98.6 & 122.94 & 0.476 & 0.349 & 181.86 & 419 \\
\hline K17 & 7051 & 107.7 & 120.49 & 0.354 & 0.314 & 181.94 & 418 \\
\hline K18 & 7099 & 105.9 & 120.38 & 0.349 & 0.304 & 184.06 & 425 \\
\hline K19 & 12161 & 107.9 & 118.00 & 0.276 & 0.281 & 183.76 & 427 \\
\hline K20 & 3369 & 99.7 & 120.39 & 0.393 & 0.279 & 183.56 & 438 \\
\hline K21 & 3370 & 99.7 & 120.36 & 0.385 & 0.272 & 184.36 & 437 \\
\hline K22 & 7161 & 108.3 & 119.16 & 0.291 & 0.271 & 183.13 & 427 \\
\hline K23 & 7146 & 108.4 & 119.32 & 0.283 & 0.282 & 181.80 & 427 \\
\hline
\end{tabular}
\end{table}

\begin{table}
\captionsetup{labelformat=empty}
\caption{Table S3. Comparison of model predictions of $\mathbf{C O}_{\mathbf{2}}$ capture efficiency with data values calculated from the liquid and gas side mass balances.}
\begin{tabular}{|l|l|l|l|l|}
\hline & \multicolumn{2}{|c|}{Experimental Data} & \multicolumn{2}{|c|}{Model Predictions} \\
\hline Case No. & Liquid Side & Gas Side & Original Model & Model with Composition Uncertainty \\
\hline K1 & 89.45 & 99.91 & 99.96 & 99.95 \\
\hline K2 & 93.26 & 99.49 & 99.84 & 99.78 \\
\hline K3 & 72.90 & 83.57 & 77.33 & 76.46 \\
\hline K4 & 70.86 & 78.10 & 76.50 & 75.72 \\
\hline K5 & 90.69 & 99.53 & 99.98 & 99.98 \\
\hline K6 & 58.89 & 59.03 & 68.16 & 60.59 \\
\hline K7 & 57.50 & 54.76 & 69.82 & 57.67 \\
\hline K8 & 93.92 & 98.07 & 99.95 & 99.95 \\
\hline K9 & 52.30 & 55.48 & 57.55 & 54.85 \\
\hline K10 & 89.69 & 98.43 & 95.35 & 94.69 \\
\hline K11 & 91.63 & 99.75 & 99.93 & 99.92 \\
\hline K12 & 92.47 & 99.61 & 99.93 & 99.92 \\
\hline K13 & 89.84 & 97.98 & 98.36 & 98.11 \\
\hline K14 & 92.89 & 98.27 & 99.32 & 98.88 \\
\hline K15 & 91.72 & 99.42 & 99.44 & 99.11 \\
\hline K16 & 84.74 & 93.54 & 87.54 & 86.06 \\
\hline K17 & 88.92 & 97.61 & 98.18 & 97.88 \\
\hline K18 & 85.72 & 92.85 & 94.36 & 93.71 \\
\hline K19 & 93.18 & 98.21 & 98.91 & 98.73 \\
\hline K20 & 89.92 & 95.55 & 98.02 & 97.81 \\
\hline K21 & 90.05 & 96.32 & 99.72 & 99.69 \\
\hline K22 & 94.58 & 99.49 & 99.97 & 99.97 \\
\hline K23 & 92.61 & 99.58 & 99.96 & 99.96 \\
\hline
\end{tabular}
\end{table}

\begin{table}
\captionsetup{labelformat=empty}
\caption{Table S4. Comparison of model predictions of $\mathbf{C O}_{\mathbf{2}}$ loading in the outlet lean solvent from the regenerator}
\begin{tabular}{|l|l|l|l|l|}
\hline \multirow[b]{2}{*}{Case No.} & \multicolumn{2}{|c|}{Original} & \multicolumn{2}{|c|}{With Composition Discrepancy} \\
\hline & Data & Model & Data & Model \\
\hline K1 & 0.145 & 0.169 & 0.152 & 0.170 \\
\hline K2 & 0.247 & 0.273 & 0.260 & 0.279 \\
\hline K3 & 0.091 & 0.086 & 0.096 & 0.086 \\
\hline K4 & 0.083 & 0.090 & 0.087 & 0.090 \\
\hline K5 & 0.108 & 0.116 & 0.113 & 0.116 \\
\hline K6 & 0.347 & 0.371 & 0.365 & 0.372 \\
\hline K7 & 0.399 & 0.425 & 0.419 & 0.427 \\
\hline K8 & 0.154 & 0.165 & 0.162 & 0.166 \\
\hline K9 & 0.239 & 0.243 & 0.251 & 0.243 \\
\hline K10 & 0.062 & 0.059 & 0.065 & 0.059 \\
\hline K11 & 0.161 & 0.166 & 0.169 & 0.167 \\
\hline K12 & 0.160 & 0.164 & 0.168 & 0.165 \\
\hline K13 & 0.164 & 0.173 & 0.172 & 0.174 \\
\hline K14 & 0.224 & 0.240 & 0.235 & 0.243 \\
\hline K15 & 0.224 & 0.244 & 0.235 & 0.247 \\
\hline K16 & 0.124 & 0.119 & 0.13 & 0.119 \\
\hline K17 & 0.168 & 0.173 & 0.177 & 0.174 \\
\hline K18 & 0.141 & 0.168 & 0.148 & 0.169 \\
\hline K19 & 0.184 & 0.215 & 0.193 & 0.219 \\
\hline K20 & 0.075 & 0.084 & 0.079 & 0.084 \\
\hline K21 & 0.074 & 0.078 & 0.078 & 0.078 \\
\hline K22 & 0.130 & 0.153 & 0.137 & 0.153 \\
\hline K23 & 0.138 & 0.156 & 0.145 & 0.156 \\
\hline
\end{tabular}
\end{table}

\begin{figure}
\includegraphics[alt={},max width=\textwidth]{https://cdn.mathpix.com/cropped/8571ff82-215f-48cb-a23a-658c0dbf8e90-06.jpg?height=948&width=1539&top_left_y=249&top_left_x=294}
\captionsetup{labelformat=empty}
\caption{Figure S1: Comparison of model and data absorber temperature profiles for Cases K1-K6.}
\end{figure}

\begin{figure}
\includegraphics[alt={},max width=\textwidth]{https://cdn.mathpix.com/cropped/8571ff82-215f-48cb-a23a-658c0dbf8e90-07.jpg?height=991&width=1545&top_left_y=253&top_left_x=275}
\captionsetup{labelformat=empty}
\caption{Figure S2: Comparison of model and data absorber temperature profiles for Cases K7K12.}
\end{figure}

\begin{figure}
\includegraphics[alt={},max width=\textwidth]{https://cdn.mathpix.com/cropped/8571ff82-215f-48cb-a23a-658c0dbf8e90-08.jpg?height=948&width=1509&top_left_y=253&top_left_x=309}
\captionsetup{labelformat=empty}
\caption{Figure S3: Comparison of model and data absorber temperature profiles for Cases K13K18.}
\end{figure}

\begin{figure}
\includegraphics[alt={},max width=\textwidth]{https://cdn.mathpix.com/cropped/8571ff82-215f-48cb-a23a-658c0dbf8e90-09.jpg?height=804&width=1567&top_left_y=251&top_left_x=300}
\captionsetup{labelformat=empty}
\caption{Figure S4: Comparison of model and data absorber temperature profiles for Cases K19K23.}
\end{figure}

\begin{figure}
\includegraphics[alt={},max width=\textwidth]{https://cdn.mathpix.com/cropped/8571ff82-215f-48cb-a23a-658c0dbf8e90-10.jpg?height=957&width=1561&top_left_y=251&top_left_x=268}
\captionsetup{labelformat=empty}
\caption{Figure S5: Comparison of model and data absorber temperature profiles for Cases K1-K6.}
\end{figure}

\begin{figure}
\includegraphics[alt={},max width=\textwidth]{https://cdn.mathpix.com/cropped/8571ff82-215f-48cb-a23a-658c0dbf8e90-11.jpg?height=989&width=1575&top_left_y=234&top_left_x=277}
\captionsetup{labelformat=empty}
\caption{Figure S6: Comparison of model and data absorber temperature profiles for Cases K7K12.}
\end{figure}

\begin{figure}
\includegraphics[alt={},max width=\textwidth]{https://cdn.mathpix.com/cropped/8571ff82-215f-48cb-a23a-658c0dbf8e90-12.jpg?height=955&width=1520&top_left_y=399&top_left_x=283}
\captionsetup{labelformat=empty}
\caption{Figure S7: Comparison of model and data absorber temperature profiles for Cases K7K12.}
\end{figure}

\begin{figure}
\includegraphics[alt={},max width=\textwidth]{https://cdn.mathpix.com/cropped/8571ff82-215f-48cb-a23a-658c0dbf8e90-13.jpg?height=854&width=1584&top_left_y=253&top_left_x=277}
\captionsetup{labelformat=empty}
\caption{Figure S8: Comparison of model and data absorber temperature profiles for Cases K7K12.}
\end{figure}