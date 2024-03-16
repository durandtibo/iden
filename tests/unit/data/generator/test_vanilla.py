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
