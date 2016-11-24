#!/bin/bash

# Copyright 2016 Intel Corporation
#
# This file is part of brainiak-extras.
#
# brainiak-extras is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# brainiak-extras is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with brainiak-extras.  If not, see <http://www.gnu.org/licenses/>.

set -ev
set -o pipefail

flake8 --config setup.cfg brainiak_extras
rst-lint *.rst | { grep -v "is clean.$" || true; }
