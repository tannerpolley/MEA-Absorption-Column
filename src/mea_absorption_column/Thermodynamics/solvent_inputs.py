"""Identified MEA declarations for numerical reproduction, not solvent qualification."""
import hashlib
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]


class SolventRefusal(ValueError):
    kind = 'preflight_refusal'


class StateDomainFailure(ValueError):
    kind = 'state_domain_failure'


class NativeEvaluationFailure(RuntimeError):
    kind = 'native_evaluation_failure'


def read_json(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise SolventRefusal(f'Duplicate declaration: {key}')
            result[key] = value
        return result
    def finite_number(value):
        result = float(value)
        if not math.isfinite(result):
            raise SolventRefusal(f'Nonfinite {value}')
        return result
    return json.loads(Path(path).read_text(), object_pairs_hook=unique,
                      parse_float=finite_number, parse_constant=finite_number)


def formula_masses(declaration, ids, mapping):
    masses = [math.fsum(declaration['atomic_masses_kg_mol'][e] * n
                       for e, n in declaration['formulas'][s].items()) for s in ids]
    native = {c['component_id']: c['fixed']['molar_mass']['value'] for c in mapping['components']}
    if any(native[s]['unit'] not in ('kilogram / mole', 'kg / mol')
           or not math.isfinite(native[s]['magnitude']) or not math.isfinite(m) or m <= 0 or abs(m - native[s]['magnitude']) > 1e-5
           for s, m in zip(ids, masses)):
        raise SolventRefusal('Formula/native molar masses disagree (absolute allowance 1e-5 kg/mol)')
    return masses


class SolventInputs:
    def __init__(self, thermodynamics, transport):
        try:
            self.thermo, self.transport = read_json(thermodynamics), read_json(transport)
            linked = Path(transport).parent / self.transport['thermodynamic_input']
            if linked.resolve() != Path(thermodynamics).resolve():
                raise SolventRefusal('Transport thermodynamic input path disagrees')
            self.hashes = {str(Path(p).resolve()): hashlib.sha256(Path(p).read_bytes()).hexdigest()
                           for p in (thermodynamics, transport)}
            if self.transport['thermodynamic_input_sha256'] != self.hashes[str(Path(thermodynamics).resolve())]:
                raise SolventRefusal('Transport belongs to another thermodynamic input')
            self.paths = {}
            for key in ('parameters', 'reactions', 'calorics', 'gas_parameters'):
                self.paths[key] = self.verify(self.thermo[key])
            self.case_path = self.verify(self.transport['case'])
            self.case = read_json(self.case_path)
            self.mapping, self.reactions = (read_json(self.paths[k]) for k in ('parameters', 'reactions'))
            self.roles = self.thermo['roles']
            self.species = self.thermo['species_order']
            self.feeds = self.thermo['liquid_feed_order']
            self.gas = self.thermo['vapor_feed_order']
            order = self.thermo['components']['order']
            if len(order) != len(set(order)):
                raise SolventRefusal('Duplicate analytical-component IDs')
            self.components = dict(zip(self.thermo['components']['order'], self.thermo['components']['rows'], strict=True))
            self.validate()
        except (KeyError, TypeError, ValueError, OSError) as error:
            raise SolventRefusal(f'Solvent declarations: {error}') from error

    def verify(self, record):
        if set(record) != {'repository_path', 'sha256'}:
            raise SolventRefusal('File reference requires only path and SHA-256')
        path = Path(record['repository_path'])
        path = path if path.is_absolute() else ROOT / path
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != record['sha256']:
            raise SolventRefusal(f'SHA-256 mismatch: {path}')
        self.hashes[str(path)] = digest
        return path

    def validate(self):
        t, u = self.thermo, self.transport
        fields = "encoding status engine parameters reactions calorics gas_parameters species_order roles liquid_support liquid_feed_order vapor_support vapor_feed_order neutral_reference species_reference_encoding components charge_numbers formulas atomic_masses_kg_mol mass_convention qualification".split()
        transport_fields = "encoding thermodynamic_input case film kinetics enhancement hydraulic_density species_diffusivities pair_diffusivity stationary_film_components zero_charge_flux quadrature_points thickness property_laws execution_policy domain thermodynamic_input_sha256".split()
        if set(t) != set(fields) or set(u) != set(transport_fields):
            raise SolventRefusal('Missing or unknown input declarations')
        nested = [
            ('roles', t['roles'], 'co2 water amines inert_gases'),
            ('engine', t['engine'], 'engine_commit wheel_filename wheel_sha256 reactive_inputs_sha256 release_tag'),
            ('engine.reactive_inputs_sha256', t['engine']['reactive_inputs_sha256'], 'parameters.json engine-reactions.json ideal-gas-thermochemistry.json'),
            ('components', t['components'], 'order rows'),
            ('qualification', t['qualification'], 'status assessment'),
            ('execution_policy', u['execution_policy'], 'id temperature_bounds_k void_fraction temperature_scale_k temperature_span_k initial_bulk_state initial_interface_bracket physical_residual_tolerance branch_policy'),
            ('execution_policy.branch_policy', u['execution_policy']['branch_policy'], 'loading_anchor max_log_loading_step max_loading_steps'),
            ('domain', u['domain'], 'representation temperature_k pressure_pa unloaded_amine_mass_fraction loading_mol_co2_per_mol_total_amine status predicates'),
            ('domain.pressure_pa', u['domain']['pressure_pa'], 'minimum_exclusive maximum_inclusive'),
            ('domain.loading', u['domain']['loading_mol_co2_per_mol_total_amine'], 'minimum_exclusive maximum_inclusive status'),
            ('atomic_masses_kg_mol', t['atomic_masses_kg_mol'], 'C H N O')]
        nested += [('domain.unloaded_amine_mass_fraction.' + s, row, 'kind value source')
                   for s, row in u['domain']['unloaded_amine_mass_fraction'].items()]
        for name, row, keys in nested:
            if not isinstance(row, dict) or set(row) != set(keys.split()):
                raise SolventRefusal(f'Missing or unsupported fields in {name}')
        if set(t['formulas']) != set(self.species) or any(set(row) - set(t['atomic_masses_kg_mol']) for row in t['formulas'].values()):
            raise SolventRefusal('Missing species formula or unsupported elements')
        mass_convention = {'feed': 'exact retained direct SI molar feeds; no reconversion',
                           'liquid_hydraulic': 'formula masses using atomic_masses_kg_mol',
                           'vapor_hydraulic': 'unchanged gas parameter JSON molar masses',
                           'engine': 'unchanged native parameter JSON; formula/native absolute allowance 1e-5 kg/mol',
                           'legacy_property': 'unchanged config.Constants MWs_l/MWs_v for retained named laws'}
        if t['mass_convention'] != mass_convention:
            raise SolventRefusal('Unsupported hydraulic/feed/property mass convention')
        if t['encoding'] != 'issue85-proposed-stage1-json-v1' or u['encoding'] != t['encoding']:
            raise SolventRefusal('Unsupported declaration encoding')
        identity = read_json(ROOT / 'integration/epcsaft_contract.json')['final_identity']
        if any(t['engine'][k] != identity[k] for k in ('engine_commit', 'wheel_sha256')):
            raise SolventRefusal('Solvent Engine identity differs from column pin')
        if (len(set(self.species)) != 9 or len(set(self.feeds)) != 3 or len(set(self.gas)) != 4
                or self.species != [c['component_id'] for c in self.mapping['components']]
                or set(self.feeds) != {self.roles['co2'], self.roles['water'], *self.roles['amines']}
                or self.roles['amines'] != ['monoethanolamine']
                or self.roles['inert_gases'] != ['nitrogen', 'oxygen']
                or set(t['liquid_support']) != set(self.species) or set(t['vapor_support']) != set(self.gas)):
            raise SolventRefusal('Stage 1 requires the declared nine-species MEA and four-gas topology')
        if self.roles['co2'] != 'carbon-dioxide' or self.roles['water'] != 'water':
            raise SolventRefusal('CO2/water role identities disagree with retained MEA laws')
        for ids in (self.species, self.feeds, self.gas, t['liquid_support'], t['vapor_support']):
            if len(ids) != len(set(ids)):
                raise SolventRefusal('Duplicate species, feed or support IDs')
        gas_mapping = read_json(self.paths['gas_parameters'])
        if [c['component_id'] for c in gas_mapping['components']] != self.gas:
            raise SolventRefusal('Vapor parameter/feed order disagrees')
        if (t['neutral_reference'] != self.reactions['neutral_reference'] or
                set(t['neutral_reference']) != {'solvent', 'solute_molality_ratios', 'standard_molality_mol_per_kg'}):
            raise SolventRefusal('Unsupported or inconsistent neutral reference')
        ref = t['neutral_reference']
        ratios = ref['solute_molality_ratios']
        if (ref['solvent'] != self.roles['water'] or ref['standard_molality_mol_per_kg'] <= 0
                or set(ratios) != set(self.species) - {ref['solvent']}
                or any(not math.isfinite(v) or v <= 0 for v in ratios.values())):
            raise SolventRefusal('Incomplete neutral-reference dilution path')
        z = np.asarray(t['charge_numbers'], float)
        native_z = [c['fixed']['charge_number']['value']['magnitude'] for c in self.mapping['components']]
        if set(self.components) != {'co2', 'water', *('amine:' + s for s in self.roles['amines'])}:
            raise SolventRefusal('Missing or unknown analytical components')
        b = np.asarray(list(self.components.values()), float)
        masses = formula_masses(t, self.species, self.mapping)
        if z.shape != (9,) or not np.array_equal(z, native_z) or b.shape != (3, 9) or not np.isfinite(b).all():
            raise SolventRefusal('Charge/component dimensions or native identities disagree')
        if abs(sum(z[self.species.index(s)] * v for s, v in ratios.items())) > 1e-12:
            raise SolventRefusal('Neutral-reference path is charged')
        if self.reactions['parameter_sha256'] != t['parameters']['sha256']:
            raise SolventRefusal('Reactions belong to another parameter record')
        from epcsaft import equilibrium as q
        if len({r['name'] for r in self.reactions['reactions']}) != len(self.reactions['reactions']):
            raise SolventRefusal('Duplicate reactions')
        for row in self.reactions['reactions']:
            c = row['correlation']
            if (row['reference']['source_basis'] not in q.ReferenceSourceBasis.__members__ or
                    not 0 < c['temperature_min'] <= c['temperature_max'] or not c['standard_state_id'] or
                    any(not math.isfinite(c[k]) for k in ('a', 'b', 'c', 'd', 'reference_temperature'))):
                raise SolventRefusal('Unsupported reaction correlation/reference declaration')
            if not set(row['stoichiometry']) <= set(self.species):
                raise SolventRefusal('Reaction contains an undeclared species')
            nu = np.array([row['stoichiometry'].get(s, 0.) for s in self.species])
            atoms = np.array([[t['formulas'][s].get(e, 0) for s in self.species] for e in t['atomic_masses_kg_mol']])
            if not np.isfinite(nu).all() or np.max(abs(atoms @ nu)) > 1e-12:
                raise SolventRefusal('Reaction does not conserve declared elements')
            if max(np.max(abs(b @ nu)), abs(z @ nu), abs(np.dot(masses, nu))) > 1e-12:
                raise SolventRefusal(f'Reaction conservation failed: {row["name"]}')
        if set(u['stationary_film_components']) != {'water', *('amine:' + s for s in self.roles['amines'])}:
            raise SolventRefusal('Unsupported stationary film components')
        constraints = np.array([z, *[self.components[c] for c in u['stationary_film_components']]])
        normalized = constraints / np.linalg.norm(constraints, axis=1)[:, None]
        if not np.isfinite(normalized).all() or np.linalg.matrix_rank(normalized-normalized.mean(axis=1, keepdims=True)) != len(constraints):
            raise SolventRefusal('Film constraints must be independent modulo total flux')
        expected = {'film': 'equilibrium_manifold', 'hydraulic_density': 'solved_state', 'pair_diffusivity': 'harmonic_mean',
                    'kinetics': 'not_used_in_instantaneous_equilibrium_film', 'enhancement': 'not_used', 'zero_charge_flux': True,
                    'thickness': 'bulk_CO2_diffusivity / liquid_mass_transfer_coefficient'}
        if any(u[k] != value for k, value in expected.items()):
            raise SolventRefusal('Unsupported film, kinetic, density or transport declaration')
        policy = u['execution_policy']
        physical = self.case['physical_inputs']
        if (policy['branch_policy'] != physical['liquid_branch_policy'] or
                policy['initial_bulk_state'] != self.case['initial_bulk_state'] or
                policy['initial_interface_bracket'] != self.case['initial_interface_bracket'] or
                policy['physical_residual_tolerance'] != self.case['physical_residual_tolerance']):
            raise SolventRefusal('Case initialization/continuation policy disagrees')
        if t['qualification']['status'] != 'numerical_reproduction_only; no absorber qualification supplied':
            raise SolventRefusal('Stage 1 does not qualify a solvent prediction')
        domain = u['domain']
        if (domain['temperature_k'] != [293.15, 393.15] or
                domain['pressure_pa'] != {'minimum_exclusive': 0., 'maximum_inclusive': 1e7} or
                set(domain['unloaded_amine_mass_fraction']) != set(self.roles['amines']) or
                domain['unloaded_amine_mass_fraction'][self.roles['amines'][0]]['value'] != self.case['model_input_row']['w_MEA'] or
                domain['loading_mol_co2_per_mol_total_amine']['minimum_exclusive'] != 0. or
                domain['loading_mol_co2_per_mol_total_amine']['maximum_inclusive'] is not None):
            raise SolventRefusal('Unsupported reproduction domain; no validated loading interval is asserted')
        if (policy['id'] != 'retained_3c_only' or policy['temperature_bounds_k'] != [293.15, 393.15]
                or policy['void_fraction'] != .97 or policy['temperature_scale_k'] != 300. or policy['temperature_span_k'] != 100.):
            raise SolventRefusal('Only the retained 3C numerical policy is available')
        selections = {'liquid_viscosity': ('Properties/Transport_Properties.py', 'viscosity:liquid'),
                      'gas_viscosity': ('Properties/Transport_Properties.py', 'viscosity:vapor'),
                      'bulk_diffusivity': ('Properties/Transport_Properties.py', 'diffusivity:liquid'),
                      'gas_diffusivity': ('Properties/Transport_Properties.py', 'diffusivity:vapor'),
                      'surface_tension': ('Properties/Thermophysical_Properties.py', 'surface_tension'),
                      'gas_conductivity': ('Properties/Thermophysical_Properties.py', 'thermal_conductivity_expression'),
                      'area': ('Transport/Hydraulic_Variables_Correlations.py', 'interfacial_area_expression'),
                      'holdup': ('Transport/Hydraulic_Variables_Correlations.py', 'holdup_expression'),
                      'pressure_drop': ('Transport/Pressure_Drop.py', 'pressure_drop_expression'),
                      'liquid_transfer': ('Transport/Transfer_Coefficients.py', 'liquid_mass_transfer_expression'),
                      'gas_transfer': ('Transport/Transfer_Coefficients.py', 'gas_mass_transfer_expression'),
                      'heat_transfer': ('Transport/Transfer_Coefficients.py', 'heat_transfer_expression')}
        if set(u['property_laws']) != set(selections):
            raise SolventRefusal('Incomplete property laws')
        units = {'liquid_viscosity': 'Pa s', 'gas_viscosity': 'mixture and species vector, Pa s',
                 'bulk_diffusivity': 'CO2, MEA, ion diffusivities m2/s', 'gas_diffusivity': 'CO2/water/N2/O2 m2/s',
                 'surface_tension': 'N/m', 'gas_conductivity': 'W/(m K), consumes gas species viscosities',
                 'area': 'm2/m3 bed; area per length m2/m', 'holdup': 'bed volume fraction',
                 'pressure_drop': 'Pa/m', 'liquid_transfer': 'm/s', 'gas_transfer': 'mol/(m2 s Pa)',
                 'heat_transfer': 'W/(m2 K)'}
        for key, (file, function) in selections.items():
            law = u['property_laws'][key]
            if set(law) != {"selection", "source", "coefficient_encoding", "output_units"}:
                raise SolventRefusal(f"Unsupported law fields: {key}")
            if law['output_units'] != units[key]:
                raise SolventRefusal(f'Unsupported output units: {key}')
            if law['selection'] != function or self.verify(law['source']) != ROOT / 'src/mea_absorption_column' / file:
                raise SolventRefusal(f'Unsupported law: {key}')
        if set(u['species_diffusivities']) != set(self.species):
            raise SolventRefusal('Incomplete species diffusivities')
        for species, row in u['species_diffusivities'].items():
            keys = {'model', 'prefactor_m2_s', 'activation_j_mol', 'gas_constant_j_mol_k'} if species == self.roles['co2'] else {'constant_m2_s'}
            if set(row) != keys or ('model' in row and row['model'] != 'prefactor_exp_minus_E_over_RT'):
                raise SolventRefusal('Unsupported diffusivity law or coefficient units')
            if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0
                   for k, v in row.items() if k != 'model'):
                raise SolventRefusal('Diffusivity coefficients must be finite positive')
        self.constraints = constraints

    def diffusivities(self, temperature):
        import casadi as ca
        values = []
        for species in self.species:
            row = self.transport['species_diffusivities'][species]
            values.append(row['constant_m2_s'] if 'constant_m2_s' in row else row['prefactor_m2_s'] * ca.exp(
                -row['activation_j_mol'] / (row['gas_constant_j_mol_k'] * temperature)))
        return ca.vertcat(*values)
