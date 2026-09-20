from __future__ import annotations

from clefts.ml.specgen.predict_spectrum import direct_input_dataset
from clefts.ml.specgen.source_action_msdataset import predict_source_msdataset
from clefts.ml.specgen.spectrum_generator import create_spectrum_generator
from tests.ml.action.TestActionPipeline import config


def _generator(ion_prediction_threshold=0.5):
    value = config()
    value['action_model_params']['action_prefilter_threshold_logit'] = 1.
    value['post_model_params']['ion_prediction_threshold'] = ion_prediction_threshold
    return create_spectrum_generator(value).eval()


def test_predict_source_msdataset_skips_bad_records_and_reports_them():
    dataset = direct_input_dataset(
        smiles_values=['CCCO', 'CCCO', 'CCCO'],
        collision_energy='20 eV',
        adduct_type='[M+H]+',
    )
    # Row 1 (index 1) has the exact malformed adduct from the reported bug;
    # everything else in the dataset must still get predicted.
    dataset['AdductType'] = ['[M+H]+', '[Cat]+', '[M+H]+']

    predicted, failures = predict_source_msdataset(
        dataset, _generator(),
        smiles_column='SMILES', adduct_type_column='AdductType',
        collision_energy_column='CollisionEnergy', precursor_mz_column='PrecursorMZ',
        instrument_column=None,
    )

    assert len(predicted) == 2
    assert len(failures) == 1
    assert failures.iloc[0]['index'] == 1
    assert "'Cat'" in failures.iloc[0]['error']
    assert list(predicted.metadata['SMILES']) == ['CCCO', 'CCCO']


def test_predict_source_msdataset_survives_unexpected_exception_types():
    # Adduct.parse('[M+i]+') raises AttributeError (a real, separate bug in its
    # unfinished isotope-notation handling), not ValueError/TypeError. Any
    # exception type from a bad record must be caught, not just the ones this
    # function anticipated.
    dataset = direct_input_dataset(
        smiles_values=['CCCO', 'CCCO'],
        collision_energy='20 eV',
        adduct_type='[M+H]+',
    )
    dataset['AdductType'] = ['[M+H]+', '[M+i]+']

    predicted, failures = predict_source_msdataset(
        dataset, _generator(),
        smiles_column='SMILES', adduct_type_column='AdductType',
        collision_energy_column='CollisionEnergy', precursor_mz_column='PrecursorMZ',
        instrument_column=None,
    )

    assert len(predicted) == 1
    assert len(failures) == 1
    assert failures.iloc[0]['index'] == 1
    assert failures.iloc[0]['error'].startswith('AttributeError:')


def test_predict_source_msdataset_empty_failures_when_everything_is_valid():
    dataset = direct_input_dataset(
        smiles_values=['CCCO', 'CCCO'],
        collision_energy='20 eV',
        adduct_type='[M+H]+',
    )
    predicted, failures = predict_source_msdataset(
        dataset, _generator(),
        smiles_column='SMILES', adduct_type_column='AdductType',
        collision_energy_column='CollisionEnergy', precursor_mz_column='PrecursorMZ',
        instrument_column=None,
    )
    assert len(predicted) == 2
    assert failures.empty
    assert list(failures.columns) == ['index', 'smiles', 'error']


def test_predict_source_msdataset_adds_main_adduct_and_model_collision_energy_columns():
    # [M+H]+ is already one of the model's reference adduct types, so the
    # loss-stripped MainAdduct is unchanged; ModelCollisionEnergyEV is the
    # numeric eV value actually fed to the model, not the raw column text.
    dataset = direct_input_dataset(
        smiles_values=['CCCO'],
        collision_energy='20 eV',
        adduct_type='[M+H]+',
    )
    predicted, failures = predict_source_msdataset(
        dataset, _generator(),
        smiles_column='SMILES', adduct_type_column='AdductType',
        collision_energy_column='CollisionEnergy', precursor_mz_column='PrecursorMZ',
        instrument_column=None,
    )
    assert failures.empty
    assert predicted.metadata['MainAdduct'].tolist() == ['[M+H]+']
    assert predicted.metadata['ModelCollisionEnergyEV'].tolist() == [20.0]


def test_predict_source_msdataset_ion_prediction_threshold_filters_low_confidence_peaks():
    dataset = direct_input_dataset(
        smiles_values=['CCCO'],
        collision_energy='20 eV',
        adduct_type='[M+H]+',
    )
    # An untrained model's ion confidence is effectively arbitrary, but it is
    # always within (0, 1); a threshold above every possible value must drop
    # every peak without crashing (an all-peaks-filtered spectrum used to
    # break MSDataset construction), and a threshold below every possible
    # value must keep peaks flowing through unfiltered.
    strict, _ = predict_source_msdataset(
        dataset, _generator(ion_prediction_threshold=0.999999),
        smiles_column='SMILES', adduct_type_column='AdductType',
        collision_energy_column='CollisionEnergy', precursor_mz_column='PrecursorMZ',
        instrument_column=None,
    )
    assert strict.n_peaks_total == 0

    lenient, _ = predict_source_msdataset(
        dataset, _generator(ion_prediction_threshold=0.000001),
        smiles_column='SMILES', adduct_type_column='AdductType',
        collision_energy_column='CollisionEnergy', precursor_mz_column='PrecursorMZ',
        instrument_column=None,
    )
    assert lenient.n_peaks_total > 0
