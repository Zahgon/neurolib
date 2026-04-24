import os
import numpy as np
import matplotlib.pyplot as plt

import nibabel as nib
import mne
from mne.datasets import eegbci
from mne.datasets import fetch_fsaverage

import logging
from xml.etree import ElementTree
from neurolib.utils.atlases import AutomatedAnatomicalParcellation2


class LeadfieldGenerator:

    """
    Authors: Mohammad Orabe <orabe.mhd@gmail.com>
             Zixuan liu <zixuan.liu@campus.tu-berlin.de> 

    A class to compute the lead-field matrix and perform related operations.
    The default loaded data is the template data 'fsaverage'.
    To establish an AAL2 atlas source space, the average dipole value within each atlas annotation is computed, a process referred to as downsampling.
    The initial step is to generate the surface source model.
    The downsampling process need NIfTI file and XML file of the AAL2 atlas.


    Parameters:
    ==========
        fs_dir (str): Path to the downloaded 'fsaverage' directory, set as default data.
        subject (str): The name of the subject.
        subjects_dir (str): Path to the directory containing the subject data.
        trans (str): Path to the coregistration transformation file.
        atlas_nii (str): Path to the NIfTI file of the atlas.
        atlas_xml (str): Path to the XML file of the atlas.

    Attributes:
    ==========
        raw (mne.io.Raw): The raw EEG data.


    Methods:
    =======
        load_data(subject, subjects_dir):
            Load subject data and its directory, 'fsaverage' is set as default. For user-specific data, a coregistration transformation file needed to be generated.

        load_transformation file(trans):
            load the transformation file of the subject, 'fsaverage' has default transformation file.

        build_BEM(subject, conductivity, subjects_dir)
            Construct BEM for the given subject head model.

        generate_surface_source_space(subject, spacing, add_dist):
            Generate the overall surface source model.

        EEG_coregistration(subject, configuration, src, trans, visualization):
            Align the selected EEG configuration with the subject and visualization.

        calculate_general_forward_solution(raw, trans, src, bem, eeg, mindist, n_jobs):
            Compute the general forward solution based on given subject, BEM, and EEG configuration.

        downsample_leadfield_matrix(leadfield, label_codes, atlas_nii, atlas_xml):
            Downsample the lead-field matrix according to AAL2 atlas based on general forward solution.

        check_atlas_missing_regions():
            Check for missing regions in the atlas based on label codes.
    """

    def __init__(self, subject):
        self.subject = subject
        self.fs_dir = None
        self.subjects_dir = None
        self.trans = None

    def load_data(self, subjects_dir=None, subject="fsaverage"):
        """
        Load subject data.

        Parameters:
        ==========
        subject (str): The name of the subject, default set as 'fsaverage'.
        subjects_dir (str): The directory of the subject.

        """
        pass

        # (raw_fname,) = eegbci.load_data(subject=1, runs=[6])
        # raw = mne.io.read_raw_edf(raw_fname, preload=True)

    def load_transformation_file(self, trans_path, subject="fsaverage"):
        """
        Load transformation file.

        Parameters:
        ==========
        trans_path (str): The directory of the transformation file

        """
        pass

    def build_BEM(
        self,
        conductivity=(0.3, 0.006, 0.3),
        visualization=True,
        brain_surfaces="white",
        orientation="coronal",
        slices=[50, 100, 150, 200],
    ):
        """
        Create the Boundary Element Model (BEM) solution for the given subject using on the linear collocation approach.

        Parameters:
        ==========
            subject (ndarray | str): Subject identifier.
            subjects_dir (str): Subject directory path.
            fs_dir (str): FreeSurfer directory path.
            conductivity : array of int, shape (3,) or (1,). The conductivities to use for each shell. Should be a single element for a one-layer model, or three elements for a three-layer model. Defaults to ``[0.3, 0.006, 0.3]``. The MNE-C default for a single-layer model would be ``[0.3]``.

        Returns:
        =======
            mne.bem.ConductorModel: BEM of the given head model.
            plot_bem_kwargs: Image information of the given mri data

        """
        pass

    def generate_surface_source_space(self, plot_bem_kwargs, spacing="ico4", add_dist="patch", visualization=True):
        """
        Generate the overall surface source model.

        Parameters:
        ==========
            subject (ndarray | str): Subject identifier.
            subjects_dir (str): Subject directory path.
            spacing (str) : The spacing to use. Can be 'ico#' for a recursively subdivided icosahedron, 'oct#' for a recursively subdivided octahedron, 'all' for all points, or an integer to use approximate distance-based spacing (in mm).
            add_dist (bool | str): Add distance and patch information to the source space.

        Returns:
        =======
            src (mne.SourceSpaces): Surface source space object.

        """
        pass

    def EEG_coregistration(self, src, configuration="standard_1020", visualization=True):
        """
        Align the selected EEG configuration with the subject and visualization.

        Parameters:
        ==========
            src (mne.SourceSpaces): Source space object.
            trans (str): Path to the transformation file.
            configuration (str): Type of EEG electrode layout, defaults to 'standard_1020'.

        Returns:
        =======
            raw (mne.io.Raw): Raw data coregistrated with EEG.
        """
        pass

    def calculate_general_forward_solution(self, raw, src, bem, eeg=True, mindist=5.0):
        """
        Calculate the general forward solution

        Parameters:
        ==========
            raw (mne.io.Raw): Raw data coregistrated with EEG.
            src (mne.SourceSpaces): Surface source space object.
            trans (str): Path to the transformation file.
            bem (mne.bem.ConductorModel): BEM of the given head model.

        Returns:
        =======
            fwd: The general forward solution.

        """
        pass

    def __create_label_lut(self, path: str) -> dict:
        """
        Create a lookup table that contains "anatomical acronyms" corresponding to the encodings of the regions
        specified by the used anatomical atlas. Adds an empty label for code "0" if not specified otherwise by atlas.

        Parameters:
        ==========
            path (str): Path to the XML file containing label information.

        Returns:
        =======
            dict: Dictionary with keys being the integer codes of regions and the values being anatomical acronyms.

        """
        pass

    def __get_backprojection(
        self, point_expanded: np.ndarray, affine: np.ndarray, affine_inverse: np.ndarray
    ) -> np.ndarray:
        """
        Transform MNI-mm-point into 'voxel-coordinate'.

        Parameters:
        ==========
            point_expanded (np.ndarray): First three elements are the 3D point in MNI-coordinate space (mm),
                                        last element being a 1 for the offset in transformations. `point_expanded` must have the shape of 4x1.
            affine (np.ndarray): Projects voxel-numbers to MNI coordinate space (mm). `affine` must have the shape of 4x4.
            affine_inverse (np.ndarray): Back projection from MNI space. `affine_inverse` must have the shape of 4x4.

        Returns:
        =======
            np.ndarray: The point projected back into "voxel-number-space", last element 1. Will return the shape of 4x1.

        """
        pass

    def __filter_for_regions(self, label_strings: list[str], regions: list[str]) -> list[bool]:
        """
        Create a list of bools indicating if the label_strings are in the regions list.
        This function can be used if one is only interested in a subset of regions defined by an atlas.

        Parameters:
        ==========
            label_strings (list[str]): List of labels that dipoles got assigned to.
            regions (list[str]): List of strings that are the acronyms for the regions of interest.

        Returns:
        =======
            list[bool]: List of bools indicating if each label_string is in the regions list.

        """
        pass

    def __get_labels_of_points(
        self,
        points: np.ndarray,
        nii_file: nib.Nifti1Image,
        xml_file: dict,
        atlas="aal2_cortical",
        cortex_parts="only_cortical_parts",
    ) -> tuple[list[bool], np.ndarray, list[str]]:
        """
        Gives labels of regions the points fall into.

        Parameters:
        ==========
            points (np.ndarray): Nx3 array of points defined in MNI space (mm).
            nii_file (nibabel.Nifti1Image): NIfTI file representing the anatomical atlas.
            xml_file (dict): Dictionary containing "anatomical acronyms" corresponding to the encodings of the regions.
            atlas (str): Specification of the anatomical atlas. Currently only "aal2_cortical" is supported and is set as default.
            cortex_parts (str): Specification of cortex parts, defaults to "only_cortical_parts".

        Returns:
        =======
            tuple[list[bool], np.ndarray, list[str]]: Tuple containing:
            - List of boolean values indicating if a valid assignment within the space defined by the atlas was found for each point.
            - Array of the assigned label codes for each point.
            - List of strings representing the "anatomical acronyms" of the assigned labels.

        """
        pass

    def __downsample_leadfield_matrix(
        self, leadfield: np.ndarray, label_codes: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Downsample the leadfield matrix by computing the average across all dipoles falling within specific regions. This process assumes a one-to-one correspondence between source positions and dipoles, as commonly found in a surface source space where the dipoles' orientations are aligned with the surface normals.

        Parameters:
        ==========
            leadfield (np.ndarray): Leadfield matrix. Channels x Dipoles.
            label_codes (np.ndarray): 1D array of region-labels assigned to the source locations.

        Returns:
        =======
            tuple[np.ndarray, np.ndarray]: Tuple containing:
            - Array that contains the label-codes of any region that at least one dipole was assigned to.
            - Channels x Regions leadfield matrix. The order of rows (channels) is unchanged compared to the input "leadfield",
            but the columns are sorted according to the "unique_labels" array.

        """
        pass

    def compute_downsampled_leadfield(
        self,
        fwd,
        atlas_nii_path,
        atlas_xml_path,
        atlas="aal2_cortical",
        cortex_parts="only_cortical_parts",
        path_to_save=None,
    ):
        """
        Compute the leadfield matrix.

        Parameters:
        ==========
            raw (mne.io.Raw): Raw data object.
            trans (str): Path to the transformation file.
            src (mne.SourceSpaces): Source space object.
            bem (mne.bem.ConductorModel): BEM object.
            subject (str): Subject identifier.
            atlas_nii_path (str): Path to the NIfTI file of the atlas.
            atlas_xml_path (str): Path to the XML file of the atlas.
            atlas (str): Specification of the anatomical atlas, defaults to "aal2_cortical".
            cortex_parts (str): Specification of cortex parts, defaults to "only_cortical_parts".
            path_to_save (str): Path to save the leadfield matrix as a binary file in NumPy .npy format, defaults to None.

        Returns:
        =======
            tuple[np.ndarray, mne.Forward, np.ndarray]: Tuple containing:
            - Channels x Regions leadfield matrix.
            - Forward solution object.
            - Array that contains the label-codes of any region that at least one dipole was assigned to.

        """
        pass

    def check_atlas_missing_regions(self, atlas_xml_path, unique_labels):
        """
        Investigate the missing regions of the atlas.

        Parameters:
        ==========
            atlas_xml_path (str): Path to the XML file containing label information.
            unique_labels (np.ndarray): Array containing the label-codes of any region that at least one dipole was assigned to.

        Returns:
        =======
            None

        """
        pass
