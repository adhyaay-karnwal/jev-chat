from jevchat.sampling import sample_from, top_items


def test_zero_temperature_is_argmax() -> None:
    chosen = sample_from(
        {"a": 0.1, "b": 0.7, "c": 0.2},
        temperature=0.0,
        top_p=1.0,
        rng=__import__("random").Random(0),
    )
    assert chosen == "b"


def test_top_items_order() -> None:
    assert top_items({"x": 0.2, "y": 0.5, "z": 0.3}, 2) == [("y", 0.5), ("z", 0.3)]
