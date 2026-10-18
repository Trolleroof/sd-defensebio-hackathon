"""Public mechanics of predictor submission; no modeling strategy or solution."""
from .executor import (TIMEOUT_S, CPU_LIMIT_S, MEMORY_LIMIT_BYTES,
                       MAX_INPUT_BYTES, MAX_OUTPUT_BYTES)


def agent_data(value):
    """Exclude device-temperature fields at public and predictor boundaries."""
    if isinstance(value, dict):
        return {key: agent_data(item) for key, item in value.items()
                if 'chip' not in str(key).lower()
                and 'dietemp' not in ''.join(c for c in str(key).lower() if c.isalnum())
                and key != 'estimated_skin_temp_c'}
    if isinstance(value, list):
        return [agent_data(item) for item in value]
    return value


def predictor_contract(*, campaign=False):
    return {
        'version': 'recovery-predictor-v1',
        'design_freedom': 'Invent the equations, model structure, fitting method, and uncertainty method from permitted evidence. No model family is required.',
        'submission': 'Send Python source as the code string of an edit action; evaluate, checkpoint, and submit use the current candidate. Do not print or upload forecasts separately.',
        'required_callable': 'predict(history, forecast_times_s)',
        'optional_callable': 'fit(support)',
        'inputs': {
            'support': 'List of complete permitted recording dictionaries, each with a rows list. Includes initial calibration and acquired recordings only; never development or private test future targets.',
            'history': 'List of row dictionaries through time_s <= 0 for one request. Required fields: time_s (seconds relative to cutoff), hr_bpm (beats per minute). Recorded context fields may also be present; use only supplied fields.',
            'forecast_times_s': 'Ordered list of future times in seconds relative to cutoff. Forecast each requested time exactly once in the supplied order.',
            'request_metadata': 'Public requests identify evaluations and describe available history. predict receives only history and forecast_times_s, not request IDs, roles, or condition labels. Use context actually present in history rows.',
        },
        'temperature_semantics': {
            'availability': 'Temperature is optional; use it only when recordings actually supply it. These definitions do not add a temperature channel or prediction target.',
            'temp_wrist': 'PPG-DaLiA wrist skin temperature in degC, or explicitly synthetic skin temperature generated from its reference-cohort statistics.',
            'forecast_boundary': 'Only temperature observed through the cutoff is usable for forecasting unless a separately declared task supplies a known future temperature scenario. Actual future skin measurements remain private.',
            'scoring': 'The current predictor output and score remain heart rate in bpm. Temperature forecasting requires its own declared target, reference provenance, metric, and units.',
        },
        'return_value': {
            'container': 'A list of dictionaries, one per requested time; no wrapper dictionary.',
            'required_exact_fields': ['time_s', 'mean', 'lower', 'upper'],
            'units': 'time_s in seconds; mean, lower, upper in bpm.',
            'constraints': 'Finite JSON numbers only; matching time_s; lower <= mean <= upper. Every requested forecast is mandatory; no omissions or abstentions.',
            'interval_level': 0.9,
            'one_row_example': {'time_s': 30, 'mean': 100, 'lower': 95, 'upper': 105},
            'example_scope': 'Shape illustration only; these numbers are not a prediction target or suggested model.',
        },
        'lifecycle': [
            'Each execution starts a fresh Python module from submitted source.',
            'If defined, fit(support) is called once before all requests in that execution; its return value is ignored. Store learned state in module globals or objects used by predict.',
            'predict is then called separately for each request. It must return the forecast list; the controller associates it with the request ID.',
            'Fitted state does not survive another evaluation or checkpoint. Source, candidates, permitted recordings, history, and budgets persist within the campaign.',
        ],
        'execution': {
            'runtime': 'Isolated Python with standard library; site packages disabled. No project imports, network, process forks, or private controller files. Temporary working-directory files are ephemeral.',
            'wall_timeout_s': TIMEOUT_S, 'cpu_limit_s': CPU_LIMIT_S,
            'memory_limit_bytes': MEMORY_LIMIT_BYTES,
            'memory_enforcement': 'Best effort under current macOS executor.',
            'max_code_bytes': 262144, 'max_input_bytes': MAX_INPUT_BYTES,
            'max_output_bytes': MAX_OUTPUT_BYTES,
        },
        'actions': {
            'decision': 'Every action requires nonempty evidence, hypothesis, action, expected_result strings, citing evidence already available.',
            'edit': 'Freeze source as a candidate; does not run or fit it.',
            'evaluate': 'Run and score current code on development requests; costs one development evaluation, including failed execution. Returns development metrics only.',
            'acquire': 'Use a released acquisition_menu session_id at its declared cost; adds the complete recording to permitted support. No fabricated or live experiment requests.',
            'revert': 'Select a previously recorded candidate_id; does not refund budgets.',
            **({'checkpoint': 'Privately score and freeze the current source, then advance one stage. Available in Build, Investigate, Transfer; returns a receipt, never private scores.'} if campaign else {}),
            'submit': ('Only in Sustain: freeze and privately score current source; end campaign.' if campaign else 'Freeze and privately score current source; end episode.'),
            'budgets': 'remaining and steps_remaining are authoritative; failed action attempts count toward the step limit. Budgets never reset on reversion, stage changes, or restart.',
        },
        'scoring': {
            'mae_bpm': 'Mean absolute difference between predicted mean and private reference hr_bpm at each future time, in bpm; lower is better.',
            'interval_score_90': 'For reference y and interval [L,U]: (U-L) + 20*max(L-y,0) + 20*max(y-U,0); lower is better.',
            'coverage_90': 'Fraction of reference values within [lower,upper]. Reported with interval score; wide intervals incur a width penalty.',
            'aggregation': ('Average future times within each session, sessions within each condition, then conditions equally.' if campaign else 'Average future times within each session, then sessions equally.'),
            'feedback_boundary': 'Development scores may guide revision. Private checkpoint/final scores never return to the participant. Private reference values are not passed to predictor code.',
            'invalid': 'Execution failure, missing forecasts, extra forecast fields, invalid times, or nonfinite values invalidate the result; failures cannot be dropped from the comparison.',
        },
    }
