from __future__ import annotations

from coola import objects_are_equal

from iden.data.generator import DataGenerator

###################################
#     Tests for DataGenerator     #
###################################


def test_data_generator_repr() -> None:
    assert repr(DataGenerator([1, 2, 3])).startswith("DataGenerator(")


def test_data_generator_str() -> None:
    assert str(DataGenerator([1, 2, 3])).startswith("DataGenerator(")


def test_data_generator_equal_true() -> None:
    assert DataGenerator([1, 2, 3]).equal(DataGenerator([1, 2, 3]))


def test_data_generator_equal_false_different_data() -> None:
    assert not DataGenerator([1, 2, 3]).equal(DataGenerator([]))


def test_data_generator_equal_false_different_copy() -> None:
    assert not DataGenerator([1, 2, 3]).equal(DataGenerator([1, 2, 3], copy=True))


def test_data_generator_equal_false_different_type() -> None:
    assert not DataGenerator([1, 2, 3]).equal(42)


def test_data_generator_equal_false_different_type_child() -> None:
    class Child(DataGenerator): ...

    assert not DataGenerator([1, 2, 3]).equal(Child([1, 2, 3]))


def test_data_generator_equal_nan_true() -> None:
    assert DataGenerator([1, 2, float("nan")]).equal(
        DataGenerator([1, 2, float("nan")]), equal_nan=True
    )


def test_data_generator_equal_nan_false() -> None:
    assert not DataGenerator([1, 2, float("nan")]).equal(DataGenerator([1, 2, float("nan")]))


def test_data_generator_generate() -> None:
    generator = DataGenerator([1, 2, 3])
    data = generator.generate()
    assert objects_are_equal(data, [1, 2, 3])


def test_data_generator_generate_copy_false() -> None:
    generator = DataGenerator([1, 2, 3])
    data = generator.generate()
    assert objects_are_equal(data, [1, 2, 3])
    data.append(4)
    data = generator.generate()
    assert objects_are_equal(data, [1, 2, 3, 4])


def test_data_generator_generate_copy_true() -> None:
    generator = DataGenerator([1, 2, 3], copy=True)
    data = generator.generate()
    assert objects_are_equal(data, [1, 2, 3])
    data.append(4)
    data = generator.generate()
    assert objects_are_equal(data, [1, 2, 3])
