# CountESS Configuration File Format

CountESS configuration is stored in a .INI file with a pretty
simple format.  [.INI Files](https://en.wikipedia.org/wiki/INI_file)
are simple and human readable/writable.

While CountESS is intended to be used via the GUI, the config file 
is designed to be editable if necessary and to be compatible with
revision control systems.

Configuration can also be stored as [JSON](#json-format), which is more
convenient for other programs to read and write.  CountESS chooses which format
to use from the name of the file: anything ending in `.json` is JSON, and
anything else is .INI.  The two hold the same information.

## Sections & Modules

.INI files are grouped into sections.
Each section corresponds to a node of the data flow graph, and
has the label of a node of the data flow graph.  All nodes must
have a distinct name.

    [Load FASTQ]

Each section contains key/value pairs which configure that node.
Keys are hierarchical, with subkeys separated by '.'

Keys and subkeys starting with `_` are reserved for CountESS.
Several reserved key/value pairs exist to define the plugin module being
used for this node:

    _module = countess.plugins.fastq
    _class = LoadFastqPlugin
    _version = 0.0.37

`_uuid` and `_hash` are used to keep the configuration file order stable
under name changes and to track changes to input files, respectively:

    _uuid = 0a0a9f2a6772942557ab5355d76a
    _hash = dffd6021bb2bd5b0af676290809ec3a53191dd81c7f70a4b28688a362182986f

CountESS also uses a reserved key/value pair to store graphical layout
information.  Currently this is stored as a major (flow-wards) and minor (sideways)
coordinate, each between 0 and 1000:

    _position = 55 500

## Connections

The connections between nodes are stored as reserved key/value pairs too:

    [Join on "Run"]
    _parent.0 = Read File Descriptions
    _parent.1 = Break out "Run"

Plugins are written to the file "top down" so that a node never appears in
the config file before all of its parents do.

## Plugin Configuration

All other keys are defined by the plugin modules.  Keys may not contain
`.` or `=` or other values used in the .INI format.

Values are stored in
Python [`repr()`](https://docs.python.org/3/library/functions.html#repr)
format and interpreted using
[`ast.literal_eval`](https://docs.python.org/3/library/ast.html#ast.literal_eval)
which avoids problems with special characters & also preserves types.

     min_avg_quality = 10.0
     greeting = 'Hello, World!'

Arrays of values are stored using numeric subkeys:

     files.0.filename = 'first_file.txt'
     files.1.filename = 'second_file.txt'
     files.2.filename = 'third_file.txt'

When loading, subkeys are reinstated in numeric order but any missing values
are skipped, so if you remove `files.1.filename` above everything will just work.
When re-writing config files from the GUI the array elements will be given 
contiguous numbers again.

## Per-Column Arrays

Where an array of values represents columns of a datatable, an additional `_label`
subkey records this mapping so it can be restored:

    columns.5._label = 'count'
    columns.5.sum = True
    columns.5.count = True
    columns.7._label = 'variant'
    columns.7.index = True

## Working with Revision Control

This file format should be well-behaved with revision control systems such as Git,
with some small caveats:

* Duplicate section names are not allowed:

     [New Counter 1]
     config = True

     [New Counter 1]
     config = False

  To fix this, manually rename the sections.

* Duplicate keys are not allowed, so revision control operations may end up with an
  invalid state such as:

     files.0.filename = 'Alice\'s File'
     files.0.filename = 'Bob\'s File'

  To fix this, manually renumber the files.

* The order of nodes is generally preserved, but parents are always written before 
  their children, which means that changes to the topology may lead to large changes
  in revision control.

These issues are all potentially addressable in future versions of the software.

## JSON Format

A JSON config file is a single object.  Every node of the data flow graph is
an entry in the `nodes` list, and parents are named rather than nested so the
nodes can be in any order:

    {
      "format": "countess-config",
      "version": 1,
      "countess_version": "0.1.28",
      "nodes": [
        {
          "name": "CSV Load",
          "uuid": "6cd3f5c1144152db536a2caf0f555eae",
          "plugin": {
            "module": "countess.plugins.csv",
            "class": "LoadCsvPlugin",
            "version": "0.1.28",
            "hash": "..."
          },
          "parameters": {
            "files": [{"filename": "input1.csv"}],
            "delimiter": ",",
            "header": true,
            "columns": [{"name": "thing", "type": "STRING"}]
          },
          "position": [0.3, 0.25],
          "sort": {"column": 0, "descending": false},
          "notes": "Some notes"
        }
      ]
    }

Only `format`, `version` and each node's `name` are required.  A node without a
`plugin` is one which hasn't been set up yet.  `parameters` holds the plugin's
configuration as produced by the plugin's `get_json_data()` method, and
filenames in it are relative to the directory containing the config file, as
in .INI files.  `NaN` and the infinities (which JSON can't represent) are
written as the strings `"NaN"`, `"Infinity"` and `"-Infinity"`.

Parameters which don't exist (any more) or which can't take the value they've
been given are reported as warnings and skipped when loading, in the same way
as with .INI files.
