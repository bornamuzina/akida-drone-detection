# Environments

Three, because they conflict. Not a preference -- pip cannot resolve them
together.

| | numpy | tensorflow | torch | used by |
|---|---|---|---|---|
| events | 1.26.4 | -- | 2.9.1+cpu | src/data event scripts, src/checks |
| akida | 2.1.3 | 2.19.1 | -- | src/model, Akida conversion |
| ultralytics | any | -- | own build | YOLO26n only |

The binding constraints:

* OpenEB's compiled python bindings are built against the numpy 1.x ABI.
* cnn2snn requires `tensorflow~=2.19.0`.

Everything else follows from those two.

These files list direct dependencies only. A `pip freeze` of the working
venvs would be far longer, but most of it is transitive, and the events
venv in particular has accumulated packages from unrelated work.
