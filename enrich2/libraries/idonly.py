#  Copyright 2016-2017 Alan F Rubin, Daniel C Esposito
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
#  along with Enrich2.  If not, see <http://www.gnu.org/licenses/>.


"""
Enrich2 libraries idonly module
===============================

Contains the concrete class ``IdOnlySeqLib`` which represents a sequencing
library derived from a counts file.
"""


from .seqlib import SeqLib


__all__ = [
    "IdOnlySeqLib"
]


class IdOnlySeqLib(SeqLib):
    """
    Class for counting data with non-variant identifiers and no associated
    FASTQ_ data.
    
    Class Attributes
    ----------------
    treeview_class_name :  `str`
        String used to render object in the GUI.
    
    Attributes
    ----------
    identifier_min_count : `int`
        Minimum count an Id must have to pass the filtering phase.
    
    Methods
    -------
    configure
        Configures the object from an dictionary loaded from a configuration 
        file.
    serialize
        Returns a `dict` with all configurable attributes stored that can
        be used to reconfigure a new instance.
    calculate
        Get the identifier counts from the counts file.
    
    Inherits
    --------
    :py:class:`~enrich2.libraries.seqlib.SeqLib`
    """

    treeview_class_name = "ID-only SeqLib"

    def __init__(self):
        SeqLib.__init__(self)
        self.identifier_min_count = 0
        self.add_label('identifiers')

    def configure(self, cfg):
        """
        Set up the object using the config object *cfg*, usually derived from
        a ``.json`` file.

        Parameters
        ----------
        cfg : `dict` or :py:class:`~enrich2.config.types.IdOnlySeqLibConfiguration`
            The object to configure this instance with.
        """
        from ..config.types import IdOnlySeqLibConfiguration

        if isinstance(cfg, dict):
            cfg = IdOnlySeqLibConfiguration(cfg)
        elif not isinstance(cfg, IdOnlySeqLibConfiguration):
            raise TypeError("`cfg` was neither a "
                            "IdOnlySeqLibConfiguration or dict.")

        SeqLib.configure(self, cfg)
        self.identifier_min_count = cfg.identifiers_cfg.min_count

    def serialize(self):
        """
        Format this object (and its children) as a config object suitable for
        dumping to a config file.

        Returns
        -------
        `dict`
            Attributes of this instance and that of inherited classes
            in a dictionary.
        """
        cfg = SeqLib.serialize(self)

        cfg['identifiers'] = dict()
        if self.identifier_min_count is not None \
                and self.identifier_min_count > 0:
            cfg['identifiers']['min count'] = self.identifier_min_count

        return cfg

    def calculate(self):
        """
        Get the identifier counts from the counts file.
        """
        if not self.check_store("/main/identifiers/counts"):
            if self.counts_file is not None:
                self.counts_from_file(self.counts_file)
            else:
                raise ValueError("Missing counts file [{}]".format(self.name))
            self.save_filtered_counts('identifiers',
                                      "count >= self.identifier_min_count")
