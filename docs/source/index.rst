Orcastrate Documentation
========================

.. image:: https://img.shields.io/badge/version-1.0.0-blue.svg
   :target: #
   :alt: Version

.. image:: https://img.shields.io/badge/python-3.8+-blue.svg
   :target: #
   :alt: Python Version

.. image:: https://img.shields.io/badge/license-MIT-green.svg
   :target: #
   :alt: License

Orcastrate is a production-grade development environment agent that transforms natural language requirements into fully functional development environments through intelligent planning and automated execution.

Features
--------

* **Natural Language Processing**: Convert descriptions into detailed execution plans
* **Template-Based Planning**: Intelligent selection of technology templates 
* **Multi-Tool Integration**: Docker, Git, filesystem operations, and more
* **Security-First Design**: Path traversal protection and permission validation
* **Async Architecture**: High-performance concurrent operations
* **Production Ready**: Comprehensive testing and error handling

Quick Start
-----------

.. code-block:: bash

   # Install dependencies
   pip install -r requirements.txt

   # Run the CLI
   python -m src.cli.main templates

   # Create an environment
   python -m src.cli.main create "Node.js web app with Express"

   # Run the demo
   python demo.py

Table of Contents
-----------------

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   getting_started
   user_guide
   cli_reference
   examples

.. toctree::
   :maxdepth: 2
   :caption: Developer Guide

   architecture
   development
   testing
   contributing

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/modules

.. toctree::
   :maxdepth: 2
   :caption: Additional Information

   changelog
   license

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`