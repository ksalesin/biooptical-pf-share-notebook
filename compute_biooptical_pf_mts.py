import os
import numpy as np

import drjit as dr
import mitsuba as mi

mi.set_variant('llvm_mono_polarized')
mi.set_log_level(mi.LogLevel.Info)

from drjit.llvm import Float

# Bio-optical parameters (given in Table 3 of [Chowdhary et al. 2020])
# dm = detritus & minerals
# ph = phytoplankton

exp_dm = 4.4        # Power-law size distribution exponent (dm)
exp_ph = 3.7        # Power-law size distribution exponent (ph)
min_radius = 0.01   # Minimum radius for size distribution (in um)
max_radius = 100.0  # Maximum radius for size distribution (in um)
ior_dm   = 1.15     # Refractive index, real (dm)
ior_ph   = 1.04     # Refractive index, real (ph)
ior_w    = 1.34     # Refractive index, real (water)
sigma_dm = 1.388e-5 # Scattering cross section (dm, in um^2)  
sigma_ph = 8.874e-5 # Scattering cross section (ph, in um^2)  
f_dm_lo_chl = 0.61  # Mix ratio dm : ph for low chlorophyll (0.03 mg/m^3)
f_dm_hi_chl = 0.34  # Mix ratio dm : ph for high chlorophyll (3.0 mg/m^3)
wavelength = 0.550  # Wavelength (in um)

# Calculation parameters
ngauss = 200
nmax = -1

sizes_dm = mi.load_dict({
                'type': 'powerlaw',
                'min_radius': min_radius,
                'max_radius': max_radius,
                'exponent': exp_dm,
                'gauss_points': ngauss
           })

sizes_ph = mi.load_dict({
                'type': 'powerlaw',
                'min_radius': min_radius,
                'max_radius': max_radius,
                'exponent': exp_ph,
                'gauss_points': ngauss
           })

pf_dm = mi.load_dict({
            'type': 'mie',
            'size': sizes_dm,
            'ior_med': ior_w,
            'ior_med_i': 0.0,
            'ior_sph': ior_dm,
            'ior_sph_i': 0.0,
            'nmax': nmax
        })

pf_ph = mi.load_dict({
            'type': 'mie',
            'size': sizes_ph,
            'ior_med': ior_w,
            'ior_med_i': 0.0,
            'ior_sph': ior_ph,
            'ior_sph_i': 0.0,
            'nmax': nmax
        })

def sph_dir(t, p):
    """ Map spherical to Euclidean coordinates """
    st, ct = dr.sincos(t)
    sp, cp = dr.sincos(p)
    return mi.Vector3f(cp * st, sp * st, ct)

def format_mueller(data, x):
    """ Convert Mueller matrices returned by drjit to numpy arrays in the
        desired format. """

    m = np.array(data)
    m = m[:,0,:,:]

    m11 = np.array(m[:,0,0])
    m12 = np.array(m[:,0,1]) / m11
    m33 = np.array(m[:,2,2]) / m11
    m34 = np.array(m[:,3,2]) / m11

    return np.column_stack((x, m11, m12, m33, m34))

# Create a dummy medium interaction to use for the evaluation
mei = dr.zeros(mi.MediumInteraction3f)
mei.wavelengths = wavelength
wi = mi.Vector3f(0, 0, 1)
mei.wi = -1 * wi
mei.sh_frame = mi.Frame3f(wi)

sampler = mi.load_dict({
            'type': 'independent'
         })

ctx = mi.PhaseFunctionContext(sampler, mi.TransportMode.Radiance)

# Evaluate phase functions in Mitsuba
n = 180
theta_o = dr.linspace(Float, 0.0, dr.pi, n)
phi_o = dr.linspace(Float, 0.0, 0.0, n) # Phi is irrelevant
wo = sph_dir(theta_o, phi_o)

theta_mts = np.rad2deg(np.array(theta_o))

# Calculate using Mitsuba
eval_dm = pf_dm.eval(ctx, mei, wo)
eval_ph = pf_ph.eval(ctx, mei, wo)

mueller_dm = format_mueller(eval_dm, theta_mts)
mueller_ph = format_mueller(eval_ph, theta_mts)

np.save('pf_data/pf_dm_mts.npy', mueller_dm)
np.save('pf_data/pf_ph_mts.npy', mueller_ph)

for chl in ['lo', 'hi']:
    if chl == 'lo':
        f_dm = f_dm_lo_chl
    elif chl == 'hi':
        f_dm = f_dm_hi_chl

    # Weight and mix phase functions (Eq. 9)
    weight_dm = f_dm * sigma_dm
    weight_ph = (1.0 - f_dm) * sigma_ph

    eval_mix = (weight_ph * eval_ph + weight_dm * eval_dm) / (weight_ph + weight_dm)

    mueller_mix = format_mueller(eval_mix, theta_mts)

    np.save('pf_data/pf_mix_' + chl + '_mts.npy', mueller_mix)