#!/bin/sh

[[ $(python --version) =~ 'Python 3' ]] || {
    echo "Adawat requires Python 3."
}

pytest -v
