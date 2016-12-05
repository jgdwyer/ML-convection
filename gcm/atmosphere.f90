!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!                                                                   !!
!!                   GNU General Public License                      !!
!!                                                                   !!
!! This file is part of the Flexible Modeling System (FMS).          !!
!!                                                                   !!
!! FMS is free software; you can redistribute it and/or modify       !!
!! it and are expected to follow the terms of the GNU General Public !!
!! License as published by the Free Software Foundation.             !!
!!                                                                   !!
!! FMS is distributed in the hope that it will be useful,            !!
!! but WITHOUT ANY WARRANTY; without even the implied warranty of    !!
!! MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the     !!
!! GNU General Public License for more details.                      !!
!!                                                                   !!
!! You should have received a copy of the GNU General Public License !!
!! along with FMS; if not, write to:                                 !!
!!          Free Software Foundation, Inc.                           !!
!!          59 Temple Place, Suite 330                               !!
!!          Boston, MA  02111-1307  USA                              !!
!! or see:                                                           !!
!!          http://www.gnu.org/licenses/gpl.txt                      !!
!!                                                                   !!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
module atmosphere_mod

use                  fms_mod, only: set_domain, write_version_number, &
                                    mpp_pe, mpp_root_pe, error_mesg, FATAL

! added by CW 12/04/03:
use                  fms_mod, only: stdlog, check_nml_error, close_file,&
                                    open_namelist_file, stdout
use             grid_physics, only: grid_phys_init, compute_grid_physics, &
                                    surface_temperature_forced
use         surface_flux_mod, only: surface_flux
use     vert_turb_driver_mod, only: vert_turb_driver_init,                    &
     vert_turb_driver, vert_turb_driver_end
use            vert_diff_mod, only: gcm_vert_diff_init, gcm_vert_diff_down, &
                                    gcm_vert_diff_up, gcm_vert_diff_end, &
                                    surf_diff_type
use         diag_manager_mod, only: register_diag_field, &
                                           register_static_field, send_data
use       dry_convection_mod, only: dry_convection_init, dry_convection
! end CW additions

! added by POG:
use            radiation_mod, only: radiation_init, radiation_down, radiation_up, &
                                    radiation_end


use                mixed_layer_mod, only: mixed_layer_init, mixed_layer, mixed_layer_end
use          lscale_cond_mod, only: lscale_cond_init, lscale_cond
use   dargan_bettsmiller_mod, only: dargan_bettsmiller_init, dargan_bettsmiller
use   neural_convection_mod, only: neural_convection_init, neural_convection
! end POG additions

use            constants_mod, only: grav, pi, cp_air, HLV

use           transforms_mod, only: trans_grid_to_spherical, trans_spherical_to_grid, &
                                    get_deg_lat, get_grid_boundaries, grid_domain,    &
                                    spectral_domain, get_grid_domain

use         time_manager_mod, only: time_type, set_time, get_time, &
                                    operator( + ), operator( - ), operator( < )

use     press_and_geopot_mod, only: compute_pressures_and_heights

use    spectral_dynamics_mod, only: spectral_dynamics_init, spectral_dynamics, spectral_dynamics_end, &
                                    get_num_levels, complete_init_of_tracers, get_axis_id, spectral_diagnostics, &
                                    complete_robert_filter

use          tracer_type_mod, only: tracer_type

use        field_manager_mod, only: MODEL_ATMOS

use       tracer_manager_mod, only: get_number_tracers

implicit none
private
!=================================================================================================================================

character(len=128) :: version= &
'$Id: atmosphere.f90,v 10.0 2003/10/27 23:31:04 arl Exp $'
      
character(len=128) :: tagname= &
'$Name:  $'
character(len=10), parameter :: mod_name='atmosphere'

!=================================================================================================================================

public :: atmosphere_init, atmosphere, atmosphere_end




!=================================================================================================================================

logical :: turb = .false.
logical :: ldry_convection = .false.
logical :: do_virtual = .false. ! whether virtual temp used in gcm_vert_diff
logical :: lwet_convection = .false.
logical :: neural_convection_flag = .false.
logical :: two_stream = .true.
logical :: mixed_layer_bc = .false.
real :: roughness_heat = 0.05
real :: roughness_moist = 0.05
real :: roughness_mom = 0.05

namelist/atmosphere_nml/ turb, ldry_convection, lwet_convection, neural_convection_flag, roughness_heat, two_stream, mixed_layer_bc, roughness_moist, roughness_mom, do_virtual

! end CW addition

!=================================================================================================================================

integer, parameter :: num_time_levels = 2
integer :: is, ie, js, je, num_levels, num_tracers, nhum
logical :: dry_model

real, pointer, dimension(:,:,:) :: p_half => NULL(), p_full => NULL()
real, pointer, dimension(:,:,:) :: z_half => NULL(), z_full => NULL()

type(tracer_type), allocatable, dimension(:) :: tracer_attributes
real,    pointer, dimension(:,:,:,:,:) :: grid_tracers => NULL()
real,    pointer, dimension(:,:,:    ) :: psg => NULL(), wg_full => NULL()
real,    pointer, dimension(:,:,:,:  ) :: ug => NULL(), vg => NULL(), tg => NULL()

real, allocatable, dimension(:,:    ) :: dt_psg
real, allocatable, dimension(:,:,:  ) :: dt_ug, dt_vg, dt_tg
real, allocatable, dimension(:,:,:,:) :: dt_tracers

real, allocatable, dimension(:)   :: deg_lat, rad_lat
real, allocatable, dimension(:,:) :: rad_lat_2d

! added by CW 12/8/03
real, allocatable, dimension(:,:)   ::                                        &
     t_surf,               &   ! surface temperature
     q_surf,               &   ! surface moisture
     u_surf,               &   ! surface U wind
     v_surf,               &   ! surface V wind
     rough_mom,            &   ! momentum roughness length for surface_flux
     rough_heat,           &   ! heat roughness length for surface_flux
     rough_moist,          &   ! moisture roughness length for surface_flux
     gust,                 &   ! gustiness constant
     flux_t,               &   ! surface sensible heat flux
     flux_q,               &   ! surface moisture flux
     flux_r,               &   ! surface radiation flux
     flux_u,               &   ! surface flux of zonal mom.
     flux_v,               &   ! surface flux of meridional mom.
     drag_m,               &   ! momentum drag coefficient
     drag_t,               &   ! heat drag coefficient
     drag_q,               &   ! moisture drag coefficient
     w_atm,                &   ! wind speed
     ustar,                &   ! friction velocity
     bstar,                &   ! buoyancy scale
     qstar,                &   ! moisture scale
     dhdt_surf,            &   ! d(sensible heat flux)/d(surface temp)
     dedt_surf,            &   ! d(latent heat flux)/d(surface temp)???
     dedq_surf,            &   ! d(latent heat flux)/d(surface moisture)???
     drdt_surf,            &   ! d(upward longwave)/d(surface temp)
     dhdt_atm,             &   ! d(sensible heat flux)/d(atmos.temp)
     dedq_atm,             &   ! d(latent heat flux)/d(atmospheric mixing rat.)
     dtaudv_atm,           &   ! d(stress component)/d(atmos wind)
     fracland,             &   ! fraction of land in gridbox
     rough,                &   ! roughness for vert_turb_driver
!JGD - 3/2015 - sea ice albedo model
     a_ice,                &   ! sea ice area as fraction of grid box (either 0 or 1)
     t_ml                      ! temperature of mixed layer (.ne. t_surf where ice is present)


real, allocatable, dimension(:,:,:) ::                                        &
     diff_m,               &   ! momentum diffusion coeff.
     diff_t,               &   ! temperature diffusion coeff.
     diss_heat,            &   ! heat dissipated by vertical diffusion
     flux_tr,              &   ! surface tracer flux
     non_diff_dt_ug,       &   ! zonal wind tendency except from vertical diffusion
     non_diff_dt_vg,       &   ! merid. wind tendency except from vertical diffusion
     non_diff_dt_tg,       &   ! temperature tendency except from vertical diffusion
     non_diff_dt_qg,       &   ! moisture tendency except from vertical diffusion
     conv_dt_tg,           &   ! temperature tendency from convection
     conv_dt_qg,           &   ! moisture tendency from convection
     cond_dt_tg,           &   ! temperature tendency from condensation
     cond_dt_qg,           &   ! moisture tendency from condensation
     t_intermed,           &   ! temperature before convection
     q_intermed               ! humidity before convection
!     dt_tg_spectraldamping     ! temperautre tendency from spectral damping (JGD

logical, allocatable, dimension(:,:) ::                                       &
     avail,                &   ! generate surf. flux (all true)
     land,                 &   ! land points (all false)
     coldT                     ! should precipitation be snow at this point

real, allocatable, dimension(:,:) ::                                          &
     bmflag,               &   ! which betts miller routines you're calling
     klzbs,                &   ! stored level of zero buoyancy values
     cape,                 &   ! convectively available potential energy
     cin,                  &   ! convective inhibition (this and the above are before the adjustment)
     invtau_bm_t,          &   ! temperature relaxation timescale                       
     invtau_bm_q,          &   ! humidity relaxation timescale                          
     capeflag,             &   ! a flag that says why cape=0                                
     rain,                 &   !
     snow


real, allocatable, dimension(:,:,:) ::                                        &
     t_ref,                &   ! relaxation temperature for bettsmiller scheme
     q_ref                     ! relaxation moisture for bettsmiller scheme

real, allocatable, dimension(:) ::                                            &
     sin_lat                   ! sine of latitude

integer ::                                                                    &
     id_diff_dt_ug,        &   ! zonal wind tendency from vertical diffusion
     id_diff_dt_vg,        &   ! merid. wind tendency from vertical diffusion
     id_diff_dt_tg,        &   ! temperature tendency from vertical diffusion
     id_diff_dt_qg,        &   ! moisture tendency from vertical diffusion
     id_conv_rain,         &   ! rain from convection
     id_cond_rain,         &   ! rain from condensation
     id_conv_dt_tg,        &   ! temperature tendency from convection
     id_conv_dt_qg,        &   ! moisture tendency from convection
     id_t_intermed,        &   ! temperature before convection
     id_q_intermed,        &   ! humidity before convection
     id_cond_dt_tg,        &   ! temperature tendency from condensation
     id_cond_dt_qg,        &   ! moisture tendency from condensation
     id_diss_heat,          &  ! temperature tendency from boundary layer frictional dissipation 
     id_dt_tg_spectraldamping, & ! temperautre tendency from spectral damping (JGD)
     id_dt_tg_corrections, & ! temperautre tendency from energy corrections (JGD)
     id_dt_vorg_spectraldamping ! vorticity tendency from spectral damping (JGD)
!     id_r_w1,        &   ! for bug testing
!     id_r_b1,        &   ! for bug testing
!     id_r_w2,  &
!     id_r_w3, &
!     id_r_b2, &
!     id_r_b3, &
!     id_x1, &
!     id_x2, &
!     id_y1, &
logical :: used
integer, dimension(4) :: axes

! end CW addition

! added by POG 
                                                                                
real, allocatable, dimension(:,:)   ::                                        &
     net_surf_sw_down,      &   ! net sw flux at surface
     surf_lw_down               ! downward lw flux at surface

                                                                                
! end POG additions



integer :: previous, current, future
logical :: module_is_initialized =.false.
character(len=4) :: ch_tmp1, ch_tmp2

integer         :: dt_integer
real            :: dt_real
type(time_type) :: Time_step

integer, dimension(4) :: axis_id

type(surf_diff_type) :: Tri_surf ! used by gcm_vert_diff

real, allocatable, dimension(:,:)   :: r_w1, r_w2
real, allocatable, dimension(:)     :: r_b1, r_b2, &
                                    xscale_mean, xscale_stnd, yscale_absmax

!=================================================================================================================================
contains
!=================================================================================================================================

subroutine atmosphere_init(Time_init, Time, Time_step_in)

type (time_type), intent(in)  :: Time_init, Time, Time_step_in

integer :: seconds, days
integer :: j, ierr, io
character(len=4) :: char_tmp1, char_tmp2

! added by CW 12/04/03
integer :: unit
! end CW addition

if(module_is_initialized) return

call write_version_number(version, tagname)

Time_step = Time_step_in
call get_time(Time_step, seconds, days)
dt_integer   = 86400*days + seconds
dt_real      = float(dt_integer)

! added by cw 12/04/03:
unit = open_namelist_file ()
ierr=1
do while (ierr /= 0)
  read  (unit, nml=atmosphere_nml, iostat=io, end=10)
  ierr = check_nml_error (io, 'atmosphere_nml')
enddo
10 call close_file (unit)

if ( mpp_pe() == mpp_root_pe() )   write (stdlog(), nml=atmosphere_nml)
! end CW addition

call get_number_tracers(MODEL_ATMOS, num_prog=num_tracers)
allocate (tracer_attributes(num_tracers))
call spectral_dynamics_init(Time, Time_step, previous, current, ug, vg, tg, psg, wg_full, tracer_attributes, &
                            grid_tracers, z_full, z_half, p_full, p_half, dry_model, nhum)

call get_grid_domain(is, ie, js, je)
call get_num_levels(num_levels)

allocate (r_w1(1:num_levels, 1:100))
allocate (r_w2(1:100,          1:num_levels))
allocate (r_b1(1:100))
allocate (r_b2(1:num_levels))

allocate (xscale_mean(1:num_levels))
allocate (xscale_stnd(1:num_levels))
allocate (yscale_absmax(1:num_levels))

allocate (dt_psg     (is:ie, js:je))
allocate (dt_ug      (is:ie, js:je, num_levels))
allocate (dt_vg      (is:ie, js:je, num_levels))
allocate (dt_tg      (is:ie, js:je, num_levels))
allocate (dt_tracers (is:ie, js:je, num_levels, num_tracers))

allocate (deg_lat    (       js:je))
allocate (rad_lat    (       js:je))
allocate (rad_lat_2d (is:ie, js:je))

! added by CW 12/8/03
allocate(t_surf      (is:ie, js:je))
allocate(q_surf      (is:ie, js:je))
allocate(u_surf      (is:ie, js:je))
allocate(v_surf      (is:ie, js:je))
allocate(rough_mom   (is:ie, js:je))
allocate(rough_heat  (is:ie, js:je))
allocate(rough_moist (is:ie, js:je))
allocate(gust        (is:ie, js:je))
allocate(flux_t      (is:ie, js:je))
allocate(flux_q      (is:ie, js:je))
allocate(flux_r      (is:ie, js:je))
allocate(flux_u      (is:ie, js:je))
allocate(flux_v      (is:ie, js:je))
allocate(drag_m      (is:ie, js:je))
allocate(drag_t      (is:ie, js:je))
allocate(drag_q      (is:ie, js:je))
allocate(w_atm       (is:ie, js:je))
allocate(ustar       (is:ie, js:je))
allocate(bstar       (is:ie, js:je))
allocate(qstar       (is:ie, js:je))
allocate(dhdt_surf   (is:ie, js:je))
allocate(dedt_surf   (is:ie, js:je))
allocate(dedq_surf   (is:ie, js:je))
allocate(drdt_surf   (is:ie, js:je))
allocate(dhdt_atm    (is:ie, js:je))
allocate(dedq_atm    (is:ie, js:je))
allocate(dtaudv_atm  (is:ie, js:je))
allocate(land        (is:ie, js:je))
allocate(avail       (is:ie, js:je))
allocate(fracland    (is:ie, js:je))
allocate(rough       (is:ie, js:je))
allocate(sin_lat     (       js:je))
allocate(diff_t      (is:ie, js:je, num_levels))
allocate(diff_m      (is:ie, js:je, num_levels))
allocate(diss_heat   (is:ie, js:je, num_levels))
allocate(flux_tr     (is:ie, js:je,             num_tracers))

!JGD 3/2015, ice-albedo model
allocate(a_ice       (is:ie, js:je))
allocate(t_ml        (is:ie, js:je))


allocate(non_diff_dt_ug  (is:ie, js:je, num_levels))
allocate(non_diff_dt_vg  (is:ie, js:je, num_levels))
allocate(non_diff_dt_tg  (is:ie, js:je, num_levels))
allocate(non_diff_dt_qg  (is:ie, js:je, num_levels))


! added by POG
                                                                                
allocate(net_surf_sw_down        (is:ie, js:je))
allocate(surf_lw_down            (is:ie, js:je))
allocate(conv_dt_tg  (is:ie, js:je, num_levels))
allocate(conv_dt_qg  (is:ie, js:je, num_levels))
allocate(cond_dt_tg  (is:ie, js:je, num_levels))
allocate(cond_dt_qg  (is:ie, js:je, num_levels))
allocate(t_intermed  (is:ie, js:je, num_levels))
allocate(q_intermed  (is:ie, js:je, num_levels))
                                                                                                     
allocate(coldT        (is:ie, js:je))
allocate(klzbs        (is:ie, js:je))
allocate(cape         (is:ie, js:je))
allocate(cin          (is:ie, js:je))
allocate(invtau_bm_t  (is:ie, js:je))
allocate(invtau_bm_q  (is:ie, js:je))
allocate(capeflag     (is:ie, js:je))
allocate(rain         (is:ie, js:je))
allocate(snow         (is:ie, js:je))
allocate(bmflag       (is:ie, js:je))


allocate(     t_ref  (is:ie, js:je, num_levels))
allocate(     q_ref  (is:ie, js:je, num_levels))
                                                                                                     
t_ref = 0.0; q_ref = 0.0
                                                                                                     
coldT = .false.
rain = 0.0; snow = 0.0

                                                                                
! end POG additions


land = .false. 
avail = .true.
rough_mom = roughness_mom
rough_heat = roughness_heat
rough_moist = roughness_moist
gust = 1.0 
q_surf = 0.0
u_surf = 0.0
v_surf = 0.0
fracland = 0.0 ! fraction of each gridbox that is land
flux_tr = 0.0
! end CW additions


dt_psg = 0.
dt_ug  = 0.
dt_vg  = 0.
dt_tg  = 0.
dt_tracers = 0.
!--------------------------------------------------------------------------------------------------------------------------------

call get_deg_lat(deg_lat)
do j=js,je
  rad_lat_2d(:,j) = deg_lat(j)*pi/180.
enddo

! POG addition:
if(mixed_layer_bc) then
 ! need an initial condition for the mixed layer temperature
 ! may be overwritten by restart file
 ! choose an unstable initial condition to allow moisture
 ! to quickly enter the atmosphere avoiding problems with the convection scheme
 t_surf = tg(:,:,num_levels,current)+1.0
 a_ice= 0
 t_ml = 0
 call mixed_layer_init(is, ie, js, je, num_levels, t_surf, a_ice, t_ml, get_axis_id(), Time)
else
 call grid_phys_init(get_axis_id(), Time)
endif

if(turb) then
! need to call gcm_vert_diff_init even if using gcm_vert_diff (rather than
! gcm_vert_diff_down) because the variable sphum is not initialized
! otherwise in the vert_diff module

! following sets do_conserve_energy to true; should change to using driver routine
! for vert_diff to avoid confusion
   call gcm_vert_diff_init (Tri_surf, ie-is+1, je-js+1, num_levels, .true., do_virtual)

   axes = get_axis_id()
   id_diss_heat = register_diag_field(mod_name, 'diss_heat',        &
     axes(1:3), Time, 'Temperature tendency from boundary layer frictional dissipation','K/s')
end if

! end POG addition


id_t_intermed = register_diag_field(mod_name, 'tg_before_convection',        &
     axes(1:3), Time, 'Temperature before convection called','K')
id_q_intermed = register_diag_field(mod_name, 'qg_before_convection',        &
     axes(1:3), Time, 'Humidity before convection called','kg/kg')

if(ldry_convection) call dry_convection_init(get_axis_id(), Time)

! POG addition:

call lscale_cond_init()

axes = get_axis_id()

id_cond_dt_qg = register_diag_field(mod_name, 'dt_qg_condensation',        &
     axes(1:3), Time, 'Moisture tendency from condensation','kg/kg/s')
id_cond_dt_tg = register_diag_field(mod_name, 'dt_tg_condensation',        &
     axes(1:3), Time, 'Temperature tendency from condensation','K/s')
id_cond_rain = register_diag_field(mod_name, 'condensation_rain',          &
     axes(1:2), Time, 'Rain from condensation','kg/m/m/s')

if(lwet_convection) then
   call dargan_bettsmiller_init()
   id_conv_dt_qg = register_diag_field(mod_name, 'dt_qg_convection',          &
        axes(1:3), Time, 'Moisture tendency from convection','kg/kg/s')
   id_conv_dt_tg = register_diag_field(mod_name, 'dt_tg_convection',          &
        axes(1:3), Time, 'Temperature tendency from convection','K/s')
   id_conv_rain = register_diag_field(mod_name, 'convection_rain',            &
        axes(1:2), Time, 'Rain from convection','kg/m/m/s')
endif

if(neural_convection_flag) then
   call neural_convection_init(r_w1, r_w2, r_b1, r_b2, &
                               xscale_mean,xscale_stnd,yscale_absmax)
   id_conv_dt_qg = register_diag_field(mod_name, 'dt_qg_convection',          &
        axes(1:3), Time, 'Moisture tendency from convection','kg/kg/s')
   id_conv_dt_tg = register_diag_field(mod_name, 'dt_tg_convection',          &
        axes(1:3), Time, 'Temperature tendency from convection','K/s')
   id_conv_rain = register_diag_field(mod_name, 'convection_rain',            &                  
        axes(1:2), Time, 'Rain from convection','kg/m/m/s')                                      
!   id_r_w1 = register_static_field(mod_name, 'r_w1', axes(5:6), 'r_w1','r_w1')                   
!   id_r_b1 = register_static_field(mod_name, 'r_b1', (/axes(6)/),
!   'r_b1','r_b1')                 
!   id_r_w2 = register_static_field(mod_name, 'r_w2', axes(6:7),  'r_w2','r_w2')                  
!   id_r_b2 = register_static_field(mod_name, 'r_b2', (/axes(6)/),
!   'r_b2','r_b2')                
!   id_r_w3 = register_static_field(mod_name, 'r_w3', (/axes(6), axes(5)/),
!   'r_w3','r_w3')       
!   id_r_b3 = register_static_field(mod_name, 'r_b3', (/axes(5)/),
!   'r_b3','r_b3')                
!   id_x1 = register_static_field(mod_name, 'xscale_min', (/axes(5)/),  &                         
!                                              'xscale_min','xscale_min')                         
!   id_x2 = register_static_field(mod_name, 'xscale_max', (/axes(5)/),  &                         
!                                              'xscale_max','xscale_max')                         
!   id_y1 = register_static_field(mod_name, 'yscale_absmax', (/axes(5)/), &                       
!                                        'yscale_absmax','yscale_absmax')                         
endif 


if(two_stream) call radiation_init(is, ie, js, je, num_levels, get_axis_id(), Time)
! end POG addition


if(turb) then
   call vert_turb_driver_init (ie-is+1,je-js+1,num_levels,get_axis_id(),Time)

   axes = get_axis_id()
   id_diff_dt_ug = register_diag_field(mod_name, 'dt_ug_diffusion',        &
        axes(1:3), Time, 'zonal wind tendency from diffusion','m/s^2')
   id_diff_dt_vg = register_diag_field(mod_name, 'dt_vg_diffusion',        &
        axes(1:3), Time, 'meridional wind tendency from diffusion','m/s^2')
   id_diff_dt_tg = register_diag_field(mod_name, 'dt_tg_diffusion',        &
        axes(1:3), Time, 'temperature diffusion tendency','T/s')
   id_diff_dt_qg = register_diag_field(mod_name, 'dt_qg_diffusion',        &
        axes(1:3), Time, 'moisture diffusion tendency','T/s')
endif

! JGD addition 4/10/15
id_dt_tg_spectraldamping = register_diag_field('dynamics', &
      'dt_tg_spectraldamping', axes(1:3),       Time, 'temperature tendency due to spectral damping',           'K/s')
id_dt_tg_corrections = register_diag_field('dynamics', &
      'dt_tg_corrections', axes(1:3),       Time, 'temperature tendency due to energy corrections',           'K/s')
id_dt_vorg_spectraldamping = register_diag_field('dynamics', &
      'dt_vorg_spectraldamping', axes(1:3),       Time, 'vorticity tendency due to spectral damping',           'sec**-1')
call complete_init_of_tracers(tracer_attributes, previous, current, grid_tracers)

module_is_initialized = .true.

return
end subroutine atmosphere_init

!=================================================================================================================================

subroutine atmosphere(Time)
type(time_type), intent(in) :: Time

real    :: delta_t
integer :: seconds, days, ntr
integer :: istat
integer :: i
type(time_type) :: Time_prev, Time_next

! added by POG 

real, dimension(is:ie, js:je, num_levels) :: tg_tmp, qg_tmp
real inv_cp_air 


! end POG addition


if(.not.module_is_initialized) then
  call error_mesg('atmosphere','atmosphere module is not initialized',FATAL)
endif

dt_ug  = 0.0
dt_vg  = 0.0
dt_tg  = 0.0
dt_psg = 0.0
dt_tracers = 0.0

! added by CW 12/16/2003
conv_dt_tg  = 0.0
conv_dt_qg  = 0.0
cond_dt_tg  = 0.0
cond_dt_qg  = 0.0
! end CW addition

t_intermed = 0.0
q_intermed = 0.0

call get_time(Time, seconds, days)

if(current == previous) then
  delta_t = dt_real
else
  delta_t = 2*dt_real
endif

t_intermed = tg(:,:,:,previous)
q_intermed = grid_tracers(:,:,:,previous,nhum)
if(id_t_intermed > 0) used = send_data(id_t_intermed, t_intermed, Time)
if(id_q_intermed > 0) used = send_data(id_q_intermed, q_intermed, Time)

! added by CW 12/7/03
if(ldry_convection) then
   grid_tracers(:,:,:,:,nhum) = 0.0
   call dry_convection(           Time,             tg(:,:,:,previous), &
                         p_full(:,:,:),                  p_half(:,:,:), &
                     conv_dt_tg(:,:,:) )
   dt_tg = dt_tg + conv_dt_tg
endif

! added by POG
if (lwet_convection) then
   rain = 0.0; snow = 0.0
   call dargan_bettsmiller (                            delta_t,              tg(:,:,:,previous),        &
                              grid_tracers(:,:,:,previous,nhum),                          p_full,        &
                                                         p_half,                           coldT,        &
                                                           rain,                            snow,        &
                                                     conv_dt_tg,                      conv_dt_qg,        &
                                                          q_ref,                          bmflag,        &
                                                          klzbs,                            cape,        &
                                                          cin,                             t_ref,        &
                                                          invtau_bm_t,               invtau_bm_q,        &
                                                          capeflag)
          
endif

if (neural_convection_flag) then
   rain = 0.0; snow = 0.0
   call neural_convection (                            delta_t,   tg(:,:,:,previous),        &
                              grid_tracers(:,:,:,previous,nhum),  p_full,        &
                                                         p_half,  rain,          &
                                                     conv_dt_tg,  conv_dt_qg,       &
                                                         r_w1, r_w2, r_b1, r_b2, &
                                                         xscale_mean, xscale_stnd, yscale_absmax)

endif

!(neural_convection_flag) then
!   if(id_r_w1 > 0) used = send_data(id_r_w1, r_w1, Time)
!   if(id_r_b1 > 0) used = send_data(id_r_b1, r_b1, Time)                                         
!   if(id_r_w2 > 0) used = send_data(id_r_w2, r_w2, Time)                                         
!   if(id_r_b2 > 0) used = send_data(id_r_b2, r_b2, Time)                                         
!   if(id_r_w3 > 0) used = send_data(id_r_w3, r_w3, Time)                                         
!   if(id_r_b3 > 0) used = send_data(id_r_b3, r_b3, Time)                                         
!   if(id_x1 > 0) used = send_data(id_x1, xscale_min, Time)                                       
!   if(id_x2 > 0) used = send_data(id_x2, xscale_max, Time)                                       
!   if(id_y1 > 0) used = send_data(id_y1, yscale_absmax, Time)                                    
!endif                                                                                            
                                                                                                  
if (lwet_convection .or. neural_convection_flag) then 
   tg_tmp = conv_dt_tg + tg(:,:,:,previous)
   qg_tmp = conv_dt_qg + grid_tracers(:,:,:,previous,nhum)
                                                                                                     
!  note the delta's are returned rather than the time derivatives
   conv_dt_tg = conv_dt_tg/delta_t
   conv_dt_qg = conv_dt_qg/delta_t
   rain       = rain/delta_t
                                                                                                     
   dt_tg = dt_tg + conv_dt_tg
   dt_tracers(:,:,:,nhum) = dt_tracers(:,:,:,nhum) + conv_dt_qg
                                                                                                     
   if(id_conv_dt_qg > 0) used = send_data(id_conv_dt_qg, conv_dt_qg, Time)
   if(id_conv_dt_tg > 0) used = send_data(id_conv_dt_tg, conv_dt_tg, Time)
   if(id_conv_rain > 0) used = send_data(id_conv_rain, rain, Time)
                                                                                                     
else
   tg_tmp = tg(:,:,:,previous)
   qg_tmp = grid_tracers(:,:,:,previous,nhum)

endif

rain = 0.0
call lscale_cond (         tg_tmp,                          qg_tmp,        &
                           p_full,                          p_half,        &
                            coldT,                            rain,        &
                             snow,                      cond_dt_tg,        &
                       cond_dt_qg )
                                                                                                  
cond_dt_tg = cond_dt_tg/delta_t
cond_dt_qg = cond_dt_qg/delta_t
rain       = rain/delta_t
                                                                                                  
dt_tg = dt_tg + cond_dt_tg
dt_tracers(:,:,:,nhum) = dt_tracers(:,:,:,nhum) + cond_dt_qg
                                                                                                  
if(id_cond_dt_qg > 0) used = send_data(id_cond_dt_qg, cond_dt_qg, Time)
if(id_cond_dt_tg > 0) used = send_data(id_cond_dt_tg, cond_dt_tg, Time)
if(id_cond_rain > 0) used = send_data(id_cond_rain, rain, Time)
                                                                                                     
! end pog addition



! POG addition: 

! Begin the radiation calculation by computing downward fluxes.
! This part of the calculation does not depend on the surface temperature.

if(two_stream) then
   call radiation_down(is, js, Time,                   &
                       rad_lat_2d(:,:),                &
                       p_half(:,:,:),                  &
                       tg(:,:,:,previous),             &
                       net_surf_sw_down(:,:),          &
                       surf_lw_down(:,:),              &
                       a_ice(:,:),                     &
                       t_surf(:,:))
end if




if (.not. mixed_layer_bc) then
                                                                                                         
!!$! infinite heat capacity
    t_surf = surface_temperature_forced(rad_lat_2d)
!!$! no heat capacity:
!!$   t_surf = tg(:,:,num_levels,previous)
                                                                                                         
!!$! surface temperature has same potential temp. as lowest layer:
!!$  t_surf = surface_temperature(tg(:,:,:,previous), p_full, p_half)
end if

                                                                                

call surface_flux(                                                             &
                  tg(:,:,num_levels,previous),                                 &
                                   grid_tracers(:,:,num_levels,previous,nhum), &
                  ug(:,:,num_levels,previous),                                 &
                  vg(:,:,num_levels,previous),                                 &
                       p_full(:,:,num_levels),                                 &
                       z_full(:,:,num_levels),                                 &
                     p_half(:,:,num_levels+1),                                 &
                                  t_surf(:,:),                                 &
                                  t_surf(:,:),                                 &
                                  q_surf(:,:),                                 &
                                  u_surf(:,:),                                 &
                                  v_surf(:,:),                                 &
                               rough_mom(:,:),                                 &
                              rough_heat(:,:),                                 &
                             rough_moist(:,:),                                 &
                                    gust(:,:),                                 &
                                  flux_t(:,:),                                 &
                                  flux_q(:,:),                                 &
                                  flux_r(:,:),                                 &
                                  flux_u(:,:),                                 &
                                  flux_v(:,:),                                 &
                                  drag_m(:,:),                                 &
                                  drag_t(:,:),                                 &
                                  drag_q(:,:),                                 &
                                   w_atm(:,:),                                 &
                                   ustar(:,:),                                 &
                                   bstar(:,:),                                 &
                                   qstar(:,:),                                 &
                               dhdt_surf(:,:),                                 &
                               dedt_surf(:,:),                                 &
                               dedq_surf(:,:),                                 &
                               drdt_surf(:,:),                                 &
                                dhdt_atm(:,:),                                 &
                                dedq_atm(:,:),                                 &
                              dtaudv_atm(:,:),                                 &
                                      delta_t,                                 &
                                    land(:,:),                                 &
                                   avail(:,:)  )



! Now complete the radiation calculation by computing the upward and net fluxes.

if(two_stream) then
   call radiation_up(is, js, Time,                   &
                     rad_lat_2d(:,:),                &
                     p_half(:,:,:),                  &
                     t_surf(:,:),                    &
                     tg(:,:,:,previous),             &
                     dt_tg(:,:,:))
end if


! end POG addition
 

if(turb) then

   call vert_turb_driver(            1,                              1, &
                                  Time,                 Time+Time_step, &
                               delta_t,                  fracland(:,:), &
                         p_half(:,:,:),                  p_full(:,:,:), &
                         z_half(:,:,:),                  z_full(:,:,:), &
                            ustar(:,:),                     bstar(:,:), &
                            rough(:,:),             ug(:,:,:,current ), &
                    vg(:,:,:,current ),             tg(:,:,:,current ), &
     grid_tracers(:,:,:,current,nhum),             ug(:,:,:,previous), & ! fixed time level bug 
                    vg(:,:,:,previous),             tg(:,:,:,previous), &
     grid_tracers(:,:,:,previous,nhum),                   dt_ug(:,:,:), &
                          dt_vg(:,:,:),                   dt_tg(:,:,:), &
                dt_tracers(:,:,:,nhum),                  diff_t(:,:,:), &
                         diff_m(:,:,:),                      gust(:,:)  )

!
!! Don't zero these derivatives as the surface flux depends implicitly
!! on the lowest level values
!! However it should be noted that these derivatives do not take into
!! account the change in the Monin-Obukhov coefficients, and so are not
!! very accurate.
!
!!$   dtaudv_atm = 0.0
!!$   dhdt_atm   = 0.0
!!$   dedq_atm   = 0.0

   if(.not.mixed_layer_bc) then
     call error_mesg('atmosphere','no diffusion implentation for non-mixed layer b.c.',FATAL)
   endif


! We must use gcm_vert_diff_down and _up rather than gcm_vert_diff as the surface flux
! depends implicitly on the surface values

!
! Don't want to do time splitting for the implicit diffusion step in case
! of compensation of the tendencies
!
   non_diff_dt_ug  = dt_ug
   non_diff_dt_vg  = dt_vg
   non_diff_dt_tg  = dt_tg
   non_diff_dt_qg  = dt_tracers(:,:,:,nhum)

   call gcm_vert_diff_down (1, 1,                                          &
                            delta_t,             ug(:,:,:,previous),       &
                            vg(:,:,:,previous),  tg(:,:,:,previous),       &
                            grid_tracers(:,:,:,previous,nhum),             &
                            grid_tracers(:,:,:,previous,:), diff_m(:,:,:), &
                            diff_t(:,:,:),                  p_half(:,:,:), &
                            p_full(:,:,:),                  z_full(:,:,:), &
                            flux_u(:,:),                    flux_v(:,:),   &
                            dtaudv_atm(:,:),                               &
                            flux_tr(:,:,:),                                &
                            dt_ug(:,:,:),                    dt_vg(:,:,:), &
                            dt_tg(:,:,:),          dt_tracers(:,:,:,nhum), &
                            dt_tracers(:,:,:,:),         diss_heat(:,:,:), &
                            Tri_surf)

  if(id_diss_heat > 0) used = send_data(id_diss_heat, diss_heat, Time)

!
! update surface temperature
!
   call mixed_layer(                                                       &
                              Time,                                        &
                              t_surf(:,:),                                 &
                              a_ice(:,:),                                  &
                              t_ml(:,:),                                   &
                              flux_t(:,:),                                 &
                              flux_q(:,:),                                 &
                              flux_r(:,:),                                 &
                                  dt_real,                                 & !  fixed timestep bug
                    net_surf_sw_down(:,:),                                 &
                        surf_lw_down(:,:),                                 &
                            Tri_surf,                                      &
                           dhdt_surf(:,:),                                 &
                           dedt_surf(:,:),                                 &
                           dedq_surf(:,:),                                 &
                           drdt_surf(:,:),                                 &
                            dhdt_atm(:,:),                                 &
                            dedq_atm(:,:))


   call gcm_vert_diff_up (1, 1, delta_t, Tri_surf,  &
                          dt_tg(:,:,:),     dt_tracers(:,:,:,nhum))

   if(id_diff_dt_ug > 0) used = send_data(id_diff_dt_ug, dt_ug - non_diff_dt_ug, Time)
   if(id_diff_dt_vg > 0) used = send_data(id_diff_dt_vg, dt_vg - non_diff_dt_vg, Time)
   if(id_diff_dt_tg > 0) used = send_data(id_diff_dt_tg, dt_tg - non_diff_dt_tg, Time)
   if(id_diff_dt_qg > 0) used = send_data(id_diff_dt_qg, dt_tracers(:,:,:,nhum) - non_diff_dt_qg, Time)


end if
! end CW addition

Time_next = Time + Time_step
if(previous == current) then
  future = num_time_levels + 1 - current
  Time_prev = Time
else
  future = previous
  Time_prev = Time - Time_step
endif



call spectral_dynamics(Time_prev, Time, Time_next, psg, ug, vg, tg, tracer_attributes, grid_tracers, &
                       previous, current, future, &
                       dt_psg, dt_ug, dt_vg, dt_tg, dt_tracers, wg_full, p_full, p_half, z_full, &
                       id_dt_tg_spectraldamping, id_dt_vorg_spectraldamping, id_dt_tg_corrections)

if(dry_model) then
  call compute_pressures_and_heights(tg(:,:,:,future), psg(:,:,future), z_full, z_half, p_full, p_half)
else
  call compute_pressures_and_heights( &
     tg(:,:,:,future), psg(:,:,future), z_full, z_half, p_full, p_half, grid_tracers(:,:,:,future,nhum))
endif
call complete_robert_filter(current, future, tracer_attributes, grid_tracers)

call spectral_diagnostics(Time_next, psg(:,:,future), ug(:,:,:,future), vg(:,:,:,future), &
                          tg(:,:,:,future), wg_full, grid_tracers(:,:,:,future,:))



previous = current
current  = future

return
end subroutine atmosphere

!=================================================================================================================================

subroutine atmosphere_end

if(.not.module_is_initialized) return

deallocate (dt_psg, dt_ug, dt_vg, dt_tg, dt_tracers)
deallocate (deg_lat, rad_lat, rad_lat_2d)

if(two_stream) call radiation_end

if(turb) call gcm_vert_diff_end

if(mixed_layer_bc) call mixed_layer_end(t_surf,a_ice,t_ml)

call spectral_dynamics_end(previous, current, ug, vg, tg, psg, wg_full, tracer_attributes, &
                           grid_tracers, z_full, z_half, p_full, p_half)

deallocate (tracer_attributes)

module_is_initialized = .false.

end subroutine atmosphere_end

!=================================================================================================================================

end module atmosphere_mod
