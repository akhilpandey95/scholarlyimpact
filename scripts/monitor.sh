#!/bin/bash

# This Source Code Form is subject to the terms of the MIT
# License. If a copy of the same was not distributed with this
# file, You can obtain one at
# https://github.com/akhilpandey95/scholarlyimpact/blob/master/LICENSE.

until ./extract_citations.py; do
    echo "'extract_citations.py' crashed with exit code $?. Restarting..." >&2
    sleep 1
done
