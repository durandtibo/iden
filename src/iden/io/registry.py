r"""Define the data loader registry for automatically load data based on
the extension.

This module provides a registry system that manages and dispatches
loaders based on extensions.
"""

from __future__ import annotations

__all__ = ["LoaderRegistry"]

from typing import TYPE_CHECKING, Any

from coola.utils.format import repr_indent, repr_mapping, str_indent, str_mapping

from iden.io.base import BaseLoader

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path


class LoaderRegistry(BaseLoader[Any]):
    """Registry that manages and dispatches loaders based on data type.

    Args:
        registry: Optional initial mapping of extensions to loaders. If provided,
            the registry is copied to prevent external mutations.

    Attributes:
        _registry: Internal mapping of registered types to loaders

    Example:
        Basic usage with a sequence loader:

        ```pycon
        >>> from iden.io import LoaderRegistry, JsonLoader
        >>> registry = LoaderRegistry({"json": JsonLoader()})
        >>> registry
        LoaderRegistry(
          (<class 'list'>): SequenceTransformer()
        )
        >>> registry.transform([1, 2, 3], str)
        ['1', '2', '3']

        ```
    """

    def __init__(self, registry: dict[str, BaseLoader[Any]] | None = None) -> None:
        self._registry: dict[str, BaseLoader[Any]] = registry.copy() if registry else {}

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}(\n  {repr_indent(repr_mapping(self._registry))}\n)"

    def __str__(self) -> str:
        return f"{self.__class__.__qualname__}(\n  {str_indent(str_mapping(self._registry))}\n)"

    def equal(self, other: Any, equal_nan: bool = False) -> bool:  # noqa: ARG002
        return type(other) is type(self)

    def load(self, path: Path) -> Any:
        extension = "".join(path.suffixes)[1:]
        loader = self.find_loader(extension)
        return loader.load(path)

    def register(
        self,
        extension: str,
        loader: BaseLoader[Any],
        exist_ok: bool = False,
    ) -> None:
        """Register a loader for a given data type.

        This method associates a loader instance with a specific Python type.
        When data of this type is transformed, the registered loader will be used.
        The cache is automatically cleared after registration to ensure consistency.

        Args:
            extension: The extension to register
            loader: The loader instance that handles this type
            exist_ok: If False (default), raises an error if the type is already
                registered. If True, overwrites the existing registration silently.

        Raises:
            RuntimeError: If the type is already registered and exist_ok is False

        Example:
            ```pycon
            >>> from iden.io import LoaderRegistry, JsonLoader
            >>> registry = LoaderRegistry()
            >>> registry.register("json", JsonLoader())
            >>> registry
            LoaderRegistry(
              (json): JsonLoader()
            )

            ```
        """
        if extension in self._registry and not exist_ok:
            msg = (
                f"Loader {self._registry[extension]} already registered "
                f"for extension '{extension}'. Use exist_ok=True to overwrite."
            )
            raise RuntimeError(msg)
        self._registry[extension] = loader

    def register_many(
        self,
        mapping: Mapping[str, BaseLoader[Any]],
        exist_ok: bool = False,
    ) -> None:
        """Register multiple loaders at once.

        This is a convenience method for bulk registration that internally calls
        register() for each type-loader pair.

        Args:
            mapping: Dictionary mapping Python types to loader instances
            exist_ok: If False (default), raises an error if any type is already
                registered. If True, overwrites existing registrations silently.

        Raises:
            RuntimeError: If any type is already registered and exist_ok is False

        Example:
            ```pycon
            >>> from iden.io import LoaderRegistry, JsonLoader, TextLoader
            >>> registry = LoaderRegistry()
            >>> registry.register_many({"json": JsonLoader(), "txt": TextLoader()})
            >>> registry
            LoaderRegistry(
              (json): JsonLoader()
              (txt): TextLoader()
            )

            ```
        """
        for ext, loader in mapping.items():
            self.register(ext, loader, exist_ok=exist_ok)

    def has_loader(self, extension: str) -> bool:
        """Check if a loader is explicitly registered for the given
        extension.

        Args:
            extension: The extension to check

        Returns:
            True if a loader is explicitly registered for this extension,
            False otherwise

        Example:
            ```pycon
            >>> from iden.io import LoaderRegistry, JsonLoader
            >>> registry = LoaderRegistry()
            >>> registry.register("json", JsonLoader())
            >>> registry.has_loader("json")
            True
            >>> registry.has_loader("text")
            False

            ```
        """
        return extension in self._registry

    def find_loader(self, extension: str) -> BaseLoader[Any]:
        """Find the appropriate loader for a given extension.

        Args:
            extension: The extension to find a loader for.

        Returns:
            The loader for the given file extension.

        Example:
            ```pycon
            >>> from iden.io import LoaderRegistry, JsonLoader
            >>> registry = LoaderRegistry()
            >>> registry.register("json", JsonLoader())
            >>> loader = registry.find_loader("json")
            >>> loader
            JsonLoader()

            ```
        """
        if (loader := self._registry.get(extension, None)) is not None:
            return loader
        msg = f"Incorrect extension: {extension}"
        raise ValueError(msg)
