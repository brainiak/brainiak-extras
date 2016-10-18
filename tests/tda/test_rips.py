#  Copyright 2016 Intel Corporation
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
from brainiak_extras.tda.rips import *

inf = float('inf')

def test_dist_mat_1():
    dist_mat_1 = [[0, 1, 1, 1.4], [1, 0, 1.4, 1], [1, 1.4, 0, 1], [1.4, 1, 1, 0]]
    pairs_with_dim = rips_filtration(2, inf, dist_mat_1)
    print("\nThere are %d persistence pairs: " % len(pairs_with_dim))
    for triplet in pairs_with_dim:
        print("Birth: ", triplet[0], ", Death: ", triplet[1], type(triplet[1]), ", Dimension: ", triplet[2])
    assert pairs_with_dim == [(1,1.4,1),
                                (0,1,0),
                                (0,1,0),
                                (0,1,0),
                                (1.4, inf, 2),
                                (0, inf, 0)]
# Birth:  1 , Death:  1.4 &lt;class &apos;float&apos;&gt; , Dimension:  1
# Birth:  0 , Death:  1 &lt;class &apos;int&apos;&gt; , Dimension:  0
# Birth:  0 , Death:  1 &lt;class &apos;int&apos;&gt; , Dimension:  0
# Birth:  0 , Death:  1 &lt;class &apos;int&apos;&gt; , Dimension:  0
# Birth:  1.4 , Death:  inf &lt;class &apos;float&apos;&gt; , Dimension:  2
# Birth:  0 , Death:  inf &lt;class &apos;float&apos;&gt; , Dimension:  0

