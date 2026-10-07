# 📦 Changelog 
[![conventional commits](https://img.shields.io/badge/conventional%20commits-1.0.0-yellow.svg)](https://conventionalcommits.org)
[![semantic versioning](https://img.shields.io/badge/semantic%20versioning-2.0.0-green.svg)](https://semver.org)
> All notable changes to this project will be documented in this file


## [0.8.5](https://github.com/humlab/penelope/compare/v0.8.4...v0.8.5) (2026-10-07)

### 🐛 Bug Fixes

* update humlab-penelope version from 0.8.4 to 0.8.5 in uv.lock ([011acca](https://github.com/humlab/penelope/commit/011acca73b1a6d720bc72a5faada136ad5498676))

## [0.8.4](https://github.com/humlab/penelope/compare/v0.8.3...v0.8.4) (2026-10-07)

### 🐛 Bug Fixes

* downgrade humlab-penelope version from 0.8.4 to 0.8.3 in uv.lock ([4e8f7de](https://github.com/humlab/penelope/commit/4e8f7deddc16c35cb167fcd1c4425d6bbdf4316e))
* update humlab-penelope version from 0.8.3 to 0.8.4 in pyproject.toml ([444d8cb](https://github.com/humlab/penelope/commit/444d8cb9a658a345c6b4f8c164a297b9ebc38586))
* update humlab-penelope version from 0.8.3 to 0.8.4 in uv.lock ([cbcf717](https://github.com/humlab/penelope/commit/cbcf717054d637ed2941b43cba33419c695c291c))

## [0.8.3](https://github.com/humlab/penelope/compare/v0.8.2...v0.8.3) (2026-10-07)

### 🐛 Bug Fixes

* add 'too-many-positional-arguments' to Pylint disable list ([4a50395](https://github.com/humlab/penelope/commit/4a503956fc5da38f3cd1fe8befa54c693f138acf))
* add conditional check for NLTK_DATA before downloading data ([3920012](https://github.com/humlab/penelope/commit/392001289ec833cfc95f82c1c9b511ac5ea60102))
* clean up imports and improve readability in various modules ([45e53d1](https://github.com/humlab/penelope/commit/45e53d124803e39d992cca949aefa0b6fdc647f2))
* fix broken release package  (0.8.4 yanked) ([03ca8a9](https://github.com/humlab/penelope/commit/03ca8a9f655b91fd83a37217d9e4a65ac89e5a2e))
* handle missing spaCy model gracefully in en_nlp fixture ([715ad1c](https://github.com/humlab/penelope/commit/715ad1c194aca253f4f3344054ea4e740c25cf18))
* remove flake8 from lint target in Makefile ([b607f52](https://github.com/humlab/penelope/commit/b607f52657bdc929b79bffab6662b5db1caf2ba0))
* remove unnecessary blank lines in goodness_of_fit.py ([f0fce73](https://github.com/humlab/penelope/commit/f0fce739c07c6fcb57da74d3db6f73b8d8b93ad1))
* replace .A.ravel() with .toarray().ravel() for sparse matrix compatibility ([2c07f6a](https://github.com/humlab/penelope/commit/2c07f6a7a3725f7004503cbfb5ff93cf36b4b4d2))
* replace custom extend function with dictionary unpacking for node and line options ([0c5501d](https://github.com/humlab/penelope/commit/0c5501de63706b32d2cef4ce143e8c492a7672dc))
* return None in en_nlp fixture for better handling of unavailable spaCy model ([8e36761](https://github.com/humlab/penelope/commit/8e36761768f5a37757a66549885dde0e04ed4665))
* suppress pylint warnings for possibly used before assignment in bugger.py ([38e6202](https://github.com/humlab/penelope/commit/38e62020978b647b7dc6e459c09886a7bd2f40c0))
* update aggregation method to use string "mean" for consistency ([d75315b](https://github.com/humlab/penelope/commit/d75315b8d11259cd44b59fc532ce7746bcad2cd9))
* update DataFrame indexing to use .iloc for consistency in tests ([70ec56d](https://github.com/humlab/penelope/commit/70ec56dcbe23087dff4c6862b1168a9fbf2a7349))
* update dependency versions in pyproject.toml ([9005bd0](https://github.com/humlab/penelope/commit/9005bd0b41274511c16ad5e1ec272856e8433cd9))
* update humlab-penelope version to 0.8.4 in uv.lock ([3725768](https://github.com/humlab/penelope/commit/3725768860855474411b30e8e13213502320cda1))
* update name of CustomJSTickFormatter to CustomJSTickFormatter (Bokeh >= 3.0) ([4431ed2](https://github.com/humlab/penelope/commit/4431ed2d5e63f8a7a6cb08de6a843365f10ef16f))
* update NLTK data paths and correct download syntax in post-install script ([53da2fa](https://github.com/humlab/penelope/commit/53da2fafc1bb7e239760f8b8451ea8b3e0d21e5e))
* update requirements for Python 3.11 compatibility ([b3be817](https://github.com/humlab/penelope/commit/b3be817d6478458ded9afe0d0c3a81fa0644aa7c))
* update tick formatter in plot_multiple_value_series to use CustomJSTickFormatter ([394f656](https://github.com/humlab/penelope/commit/394f656c3d1ff59dd7023bc98a2ca098ee9592ff))
* update token column handling to use bracket notation for fillna ([6d1dc5c](https://github.com/humlab/penelope/commit/6d1dc5c7a5d6deef4043bf6247698d1dba1d9f36))
* update type hints for co_occurrences and data attributes in CoOccurrenceHelper ([f4b6f3a](https://github.com/humlab/penelope/commit/f4b6f3a876255478cc45fde18327dbfa8f26c955))
* update XML parsing to use io.StringIO for diagnostics data ([974e3f8](https://github.com/humlab/penelope/commit/974e3f8a1414d8130e482a2e51abdfe1d24b58ab))

### 🧑‍💻 Code Refactoring

* enhance update_document_index_by_dicts_or_tuples for dtype handling and default value assignment ([b9ae4b0](https://github.com/humlab/penelope/commit/b9ae4b05429592493b81f531f0df9e26ea85859a))
* move non-gui logic from notebook module to common module. ([6f8c9d9](https://github.com/humlab/penelope/commit/6f8c9d9e5968c1b0e7dcc7e9b96c59c564a20dbf))
* simplify tuple unpacking in various files for improved readability ([0578085](https://github.com/humlab/penelope/commit/0578085297698092a926db23604710dd9b07f669))

## [0.4.0](https://github.com/humlab/penelope/compare/v0.3.18...v0.4.0) (2025-05-29)

### 🍕 Features

* add semantic-release configuration and GitHub Actions workflow for automated releases ([e7ac873](https://github.com/humlab/penelope/commit/e7ac873f65e1ee578b7b2b0a1b7db3df75e9e488))
* add word_exists method to check for word presence in token2id ([99ff490](https://github.com/humlab/penelope/commit/99ff490b73b8dff9315d60304d9da80ce4d1ac60))
* added merge topics to clusters ([18c75cb](https://github.com/humlab/penelope/commit/18c75cb4306637f04d89b21ef71023f9695a1e36))
* allow filter corpus by mask ([55a9179](https://github.com/humlab/penelope/commit/55a917904089474475f671b7057574c919c4e6d7))

### 🐛 Bug Fixes

* disable reportIncompatibleMethodOverride in pyright configuration ([a5c8838](https://github.com/humlab/penelope/commit/a5c883865a273c38b2fa114e7e5ecd1a67a87280))
* remove unnecessary whitespace and add type hint to normalize_by_raw_counts method ([94e3133](https://github.com/humlab/penelope/commit/94e3133ffde499cc8bb561d9379024f02e64a43e))

### 🧑‍💻 Code Refactoring

* added typings, named args ([ff6c770](https://github.com/humlab/penelope/commit/ff6c77018caf0c62df3b7013ad4a4657278d16d2))
* extracted tagged_frame_to_tokens ([7514899](https://github.com/humlab/penelope/commit/7514899b48388f535834925693d30a69feb8a65b))
* remove unused ListOfDicts import from utility module ([dfe874f](https://github.com/humlab/penelope/commit/dfe874fc9fee0a04ff775e39fcba4e46aeab8980))
* rename ([e1a7cba](https://github.com/humlab/penelope/commit/e1a7cba78b5fba396867e09db50da2230795160d))
* rename ([53cb068](https://github.com/humlab/penelope/commit/53cb068e406a3fd6617fcb770daab1e263616aa5))
* renamed class & file ([2df1c5e](https://github.com/humlab/penelope/commit/2df1c5ec011ac77def4dbc3240d16c6ed3621b59))
* renamed test cases ([82ead52](https://github.com/humlab/penelope/commit/82ead5254fbefe1df209c7a1f784b4d764eec0ca))
* renamed trends service ([1d64be1](https://github.com/humlab/penelope/commit/1d64be13664e502dba55a8bfef48815a6d98b4e0))
* streamline load_metadata function and improve document index loading logic ([eb4f92d](https://github.com/humlab/penelope/commit/eb4f92dd6c7510e0a899dcd721b2769b6e969848))
* update function signatures to use built-in types for consistency ([6bdd606](https://github.com/humlab/penelope/commit/6bdd606e21be3a26e43eec47cca62a932fedefad))
* Use typed class instead of dict ([6bbc882](https://github.com/humlab/penelope/commit/6bbc882cb596d59dae4825b89eebb3b6ddb61e18))
