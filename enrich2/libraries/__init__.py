#  Copyright 2016-2017 Alan F Rubin
#
#  This file is part of Enrich2.
#
#  Enrich2 is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  Enrich2 is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with Enrich2. If not, see <http://www.gnu.org/licenses/>.


"""
Enrich2 libraries module
========================

This module contains classes representing the different types of 
sequencing libraries that can be used in an ``Enrich2`` analysis. The classes
``SeqLib`` and ``VariantSeqLib`` represent the base abstract classes common to 
all libraries.
"""


__all__ = [
    "barcode",
    "barcodeid",
    "barcodemap",
    "barcodevariant",
    "basic",
    "idonly",
    "seqlib",
    "overlap",
    "variant"
]
