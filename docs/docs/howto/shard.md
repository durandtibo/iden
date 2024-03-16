# Shard

## How to implement a new shard?

This section explains how to extend `iden` to support more shard implementations.
`iden` has some built-in shard implementations, but it is quite easy to add additional
implementations.
A new shard can be implemented by extending the `iden.shard.BaseShard` class and implementing the
following 5 methods:

- `clear`
- `equal`
- `get_data`
- `get_uri`
- `is_cached`

Ideally, the instantiation of a shard object should be lightweight and the data should be loaded
in-memory only when the `get_data` is called.
