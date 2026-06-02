# Pull Request Template

## Description

<!-- Please include a summary of the change and which issue is fixed. Please also include relevant motivation and context. -->

## Type of Change

<!-- What type of change does this PR introduce? Check all that apply. -->

- [ ] Bug fix (non-breaking change fixing an issue)
- [ ] New feature (non-breaking change adding functionality)
- [ ] New strategy (addition of a new trading strategy)
- [ ] Breaking change (fix or feature that alters existing behavior)
- [ ] Documentation update
- [ ] Performance improvement
- [ ] Refactoring (no functional changes)

## Testing Performed

<!-- Confirm all testing steps have been completed before submitting. -->

- [ ] Ran `python run.py --strategy <name> --mode backtest` and it completed without errors
- [ ] Verified signal output matches expected contract (`signal` column: -1, 0, 1)
- [ ] Checked that no pandas imports were introduced in strategy or engine code
- [ ] All type hints are present on new/modified functions

## Checklist

<!-- Ensure the following requirements are met before merging. -->

- [ ] My code follows the project's coding standards (type hints, Google-style docstrings, polars expressions)
- [ ] I have added type hints to all new functions
- [ ] I have updated documentation (docs/ files, README) if needed
- [ ] My changes generate no new warnings
- [ ] Any new dependencies are added to requirements.txt
- [ ] I have read CONTRIBUTING.md

## Related Issues

<!-- Reference any related issues using "Closes #" or "Fixes #" -->

Closes #
