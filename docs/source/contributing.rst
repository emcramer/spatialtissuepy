============
Contributing
============

We welcome contributions to spatialtissuepy! This guide will help you
get started.


Getting Started
---------------

1. Fork the repository on GitHub
2. Clone your fork:

   .. code-block:: bash

       git clone https://github.com/YOUR_USERNAME/spatialtissuepy.git
       cd spatialtissuepy

3. Create a virtual environment:

   .. code-block:: bash

       python -m venv venv
       source venv/bin/activate  # On Windows: venv\Scripts\activate

4. Install in development mode:

   .. code-block:: bash

       pip install -e ".[dev]"


Development Workflow
--------------------

1. Create a feature branch:

   .. code-block:: bash

       git checkout -b feature/my-new-feature

2. Make your changes

3. Run tests:

   .. code-block:: bash

       pytest tests/ -v

4. Run linting:

   .. code-block:: bash

       flake8 spatialtissuepy
       black --check spatialtissuepy

5. Commit your changes:

   .. code-block:: bash

       git add .
       git commit -m "Add my new feature"

6. Push and create a pull request:

   .. code-block:: bash

       git push origin feature/my-new-feature


Code Style
----------

We follow these conventions:

- **PEP 8** for Python style
- **Black** for code formatting (line length 88)
- **NumPy-style docstrings** for documentation
- **Type hints** for all public functions

Example function:

.. code-block:: python

    def my_function(
        data: SpatialTissueData,
        radius: float = 50.0,
        normalize: bool = True
    ) -> np.ndarray:
        """
        Brief description of the function.

        Extended description if needed.

        Parameters
        ----------
        data : SpatialTissueData
            Input spatial tissue data.
        radius : float, default 50.0
            Neighborhood radius in spatial units.
        normalize : bool, default True
            If True, normalize the output.

        Returns
        -------
        np.ndarray
            Description of the return value.

        Examples
        --------
        >>> result = my_function(data, radius=30)
        >>> print(result.shape)
        (100,)

        Notes
        -----
        Additional notes about the implementation.
        """
        pass


Testing
-------

All new features should include tests:

.. code-block:: bash

    # Run all tests
    pytest tests/ -v

    # Run specific test file
    pytest tests/test_spatial.py -v

    # Run with coverage
    pytest tests/ --cov=spatialtissuepy --cov-report=html

Test files should be placed in the ``tests/`` directory with names
starting with ``test_``.


Documentation
-------------

Update documentation for new features:

1. Add docstrings to all public functions/classes
2. Update relevant user guide pages
3. Add to API reference if needed
4. Consider adding a tutorial for major features

Build documentation locally:

.. code-block:: bash

    cd docs
    make html
    # Open _build/html/index.html


Pull Request Guidelines
-----------------------

- Include a clear description of changes
- Reference any related issues
- Ensure all tests pass
- Update documentation as needed
- Follow the code style guidelines


Reporting Issues
----------------

When reporting bugs:

1. Check existing issues first
2. Include a minimal reproducible example
3. Specify your environment (OS, Python version, package versions)
4. Describe expected vs actual behavior


Feature Requests
----------------

We welcome feature requests! Please:

1. Describe the use case
2. Explain why existing features don't meet your needs
3. Suggest an API design if possible


Code of Conduct
---------------

Please be respectful and constructive in all interactions. We aim to
maintain a welcoming community for all contributors.


Questions?
----------

- Open an issue on GitHub
- Check existing documentation and tutorials
- Look at similar functions in the codebase for patterns
