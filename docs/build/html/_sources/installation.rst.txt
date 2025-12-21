Installation
============

Prerequisites
-------------

`fastLowess` supports Python 3.8 and newer.

Installing from PyPI
--------------------

The easiest way to install `fastLowess` is via pip:

.. code-block:: bash

    pip install fastLowess

Building from Source
--------------------

To build from source, you need a Rust compiler (1.75.0+) and `maturin`.

1.  **Clone the repository:**

    .. code-block:: bash

        git clone https://github.com/thisisamirv/fastLowess-py.git
        cd fastLowess-py

2.  **Create a virtual environment (optional but recommended):**

    .. code-block:: bash

        python -m venv venv
        source venv/bin/activate  # On Windows: venv\Scripts\activate

3.  **Install build dependencies:**

    .. code-block:: bash

        pip install maturin

4.  **Build and install:**

    .. code-block:: bash

        maturin develop --release

Verifying Installation
----------------------

You can verify the installation by running a simple Python command:

.. code-block:: python

    import fastLowess
    print(f"fastLowess version: {fastLowess.__version__}")
