# Supercharging OOD Detection

This repository contains the implementation for the first part of my thesis on Supercharging Out-of-Distribution (OOD) Detection.

## Structure

```markdown
- **/src/**: Contains the main source code for the application.
  - **main.py**: The entry point of the application.
  - **utils.py**: Helper functions used throughout the project.
  - **/config/**: Configuration files to manage settings.

- **/tests/**: All test scripts to validate functionality and ensure code quality.
  - **test_main.py**: Tests related to the `main.py` script.

- **/docs/**: All project documentation, including guides and explanations of features.
  - **index.md**: The starting point for documentation, covering project goals and usage.

- **requirements.txt**: A list of dependencies needed to run the application.

- **README.md**: Overview of the project, including setup, usage instructions, and more details.

- **LICENSE**: The licensing information for this project.


## Credits

This repository draws heavily from the following projects:

DEXTER (https://github.com/LinasNas/DEXTER.git) - for the core OOD detection framework.
illusory-attacks (https://github.com/LinasNas/illusory-attacks.git) - for the code used to train the agent policies.
