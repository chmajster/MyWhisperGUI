from app.ui.layout import UNKNOWN_SPEAKER, matching_segments, speaker_filter_options


def test_filter_combines_multiple_speakers_search_and_original_indexes():
    segments = [
        {'text': 'Cześć', 'speaker': 'Anna'},
        {'text': 'Cześć też', 'speaker': 'Adam'},
        {'text': 'Inny tekst', 'speaker': None},
        {'text': 'Odpowiedź', 'speaker': 'Anna'},
    ]
    assert [index for index, _ in matching_segments(segments, 'cZeŚĆ', ['speaker:Anna', 'speaker:Adam'])] == [0, 1]
    assert [index for index, _ in matching_segments(segments, '', [UNKNOWN_SPEAKER, 'speaker:Anna'])] == [0, 2, 3]
    assert [index for index, _ in matching_segments(segments, 'anna')] == [0, 3]
    assert len(matching_segments(segments)) == 4


def test_filter_names_do_not_collide_with_unknown_and_preserve_stale_selection():
    segments = [{'text': 'Test', 'speaker': 'Nieustalony'}, {'text': 'Test', 'speaker': None}]
    options = speaker_filter_options(segments, ['speaker:Dawna nazwa'])
    assert options[UNKNOWN_SPEAKER] == options['speaker:Nieustalony']
    assert 'speaker:Dawna nazwa' in options
    assert [index for index, _ in matching_segments(segments, speakers=['speaker:Nieustalony'])] == [0]
    assert [index for index, _ in matching_segments(segments, speakers=[UNKNOWN_SPEAKER])] == [1]
