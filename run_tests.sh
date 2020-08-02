#!/bin/sh

[[ $(python --version) =~ 'Python 3' ]] || {
    echo "Adawat requires Python 3."
}

python -m unittest discover -v -s ./tests -p '*_test.py'