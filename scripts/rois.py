import argparse
import inspect
import os

import EFC_learningfMRI.globals as gl
import imaging_pipelines.rois as rois


# Default layout of this project. Templates, filled with {sn}, {glm} and {H} ('L'/'R').
PATH_SURF = os.path.join(gl.baseDir, gl.surfDir, 'subj{sn}')
PATH_ROIS = os.path.join(gl.baseDir, gl.roiDir, 'subj{sn}')
PATH_MASK = os.path.join(gl.baseDir, 'glm{glm}', 'subj{sn}', 'mask.nii')
WHITE     = 'subj{sn}.{H}.white.32k.surf.gii'
PIAL      = 'subj{sn}.{H}.pial.32k.surf.gii'

# Atlas -> pairs of overlapping ROIs to exclude. Atlases not listed exclude nothing.
EXCLUDE = {
    'ROI': [(1, 2), (1, 6), (1, 7), (2, 3), (2, 4), (2, 5), (2, 7), (3, 4), (3, 5)],
}


def _make_cortical_rois(sn, glm=3, atlas_name='ROI', exclude=None,
                        path_surf=PATH_SURF, path_rois=PATH_ROIS, path_mask=PATH_MASK, 
                        white=WHITE, pial=PIAL):
    """Build one participant's cortical ROIs from the surfaces and the glm mask.

    Paths and `white`/`pial` filenames are templates filled with `{sn}`, `{glm}` and, for the
    surfaces, `{H}` ('L'/'R'); they default to this project's layout (module-level constants).
    `exclude=None` falls back to `EXCLUDE[atlas_name]`.
    """
    path_atlas = gl.atlasDir

    path_surf = path_surf.format(sn=sn)
    path_rois = path_rois.format(sn=sn)
    path_mask = path_mask.format(sn=sn, glm=glm)

    exclude   = EXCLUDE.get(atlas_name, []) if exclude is None else exclude

    white = [os.path.join(path_surf, white.format(sn=sn, H=H)) for H in ['L', 'R']]
    pial  = [os.path.join(path_surf, pial.format(sn=sn, H=H)) for H in ['L', 'R']]
    Rois  = rois.SurfRois(white, pial, path_mask, path_rois, atlas_name=atlas_name, atlas_dir=path_atlas)

    Rois.make_rois(exclude=exclude)


def _make_hemispheres(sn, glm=3, path_surf=PATH_SURF, path_rois=PATH_ROIS, path_mask=PATH_MASK,
                      white=WHITE, pial=PIAL):
    """Build one participant's hemisphere masks from the surfaces and the glm mask.

    Same templates as `_make_cortical_rois`.
    """
    path_surf = path_surf.format(sn=sn)
    path_rois = path_rois.format(sn=sn)
    path_mask = path_mask.format(sn=sn, glm=glm)

    white = [os.path.join(path_surf, white.format(sn=sn, H=H)) for H in ['L', 'R']]
    pial  = [os.path.join(path_surf, pial.format(sn=sn, H=H)) for H in ['L', 'R']]
    Rois  = rois.SurfRois(white, pial, path_mask, path_rois)

    Rois.make_hemispheres()


def make_rois(sns=gl.participants, **kwargs):
    """Build the cortical ROIs for each participant. `kwargs` go to `_make_cortical_rois`."""
    for sn in sns:
        print(f'rois, doing participant {sn}...')
        _make_cortical_rois(sn, **kwargs)


def make_hemispheres(sns=gl.participants, **kwargs):
    """Build the hemisphere masks for each participant. `kwargs` go to `_make_hemispheres`."""
    for sn in sns:
        print(f'hemispheres, doing participant {sn}...')
        _make_hemispheres(sn, **kwargs)


# Step name -> (function, per-participant function whose signature says which kwargs the step takes).
# Both steps are independent, built from the surfaces and glm mask.
FUNC = {
    'make_rois'       : (make_rois,        _make_cortical_rois),
    'make_hemispheres': (make_hemispheres, _make_hemispheres),
}


def main(what, **kwargs):
    """Run one step.

    `kwargs` are forwarded to the step (`sns=`, `glm=`, `atlas_name=`, `path_*=`, `white=`, `pial=`),
    but only the ones it accepts -- `make_hemispheres` takes no atlas.
    """
    if what is not None:
        func, func_sn = FUNC[what]                                              # select function
        accepted      = {'sns', *inspect.signature(func_sn).parameters}          # find what parameters are acceptable
        func(**{k: v for k, v in kwargs.items() if k in accepted})              # run the function


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Build cortical ROIs and hemisphere masks from the surfaces.')
    parser.add_argument('--what', default=None, choices=list(FUNC), help='which step to run (default: all)')
    parser.add_argument('--glm', type=int, default=None, help='GLM whose mask.nii bounds the ROIs (default: the step default, 3)')
    # Participant numbers only; any prefix/padding belongs in the templates, e.g. 's{sn:02d}'.
    parser.add_argument('--sns', nargs='+', type=int, help='participant numbers (default: all participants)')
    parser.add_argument('--atlas_name', default=None, help='atlas, rois_cortical only (default: the step default, ROI)')

    # Path templates, filled with {sn}, {glm} and {H} ('L'/'R'). Default: the module-level constants.
    parser.add_argument('--path_surf', default=None, help=f'surface folder template (default: {PATH_SURF})')
    parser.add_argument('--path_rois', default=None, help=f'output ROI folder template (default: {PATH_ROIS})')
    parser.add_argument('--path_mask', default=None, help=f'glm mask template (default: {PATH_MASK})')
    parser.add_argument('--white',     default=None, help=f'white surface filename template (default: {WHITE})')
    parser.add_argument('--pial',      default=None, help=f'pial surface filename template (default: {PIAL})')
    args = parser.parse_args()

    kwargs = {k: v for k, v in vars(args).items() if k != 'what' and v is not None}
    main(args.what, **kwargs)

    if args.what is None:
        main('make_rois',        **kwargs)
        main('make_hemispheres', **kwargs)
