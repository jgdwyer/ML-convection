
module neural_convection_mod


!----------------------------------------------------------------------
use            fms_mod, only:  file_exist, error_mesg, open_file,  &
                               check_nml_error, mpp_pe, FATAL,  &
                               close_file
use simple_sat_vapor_pres_mod, only:  escomp, descomp
use      constants_mod, only:  HLv,HLs,Cp_air,Grav,rdgas,rvgas, &
                               kappa
use netcdf

implicit none
private
!---------------------------------------------------------------------
!  ---- public interfaces ----

   public  neural_convection, neural_convection_init, check

!-----------------------------------------------------------------------
!   ---- version number ----

 character(len=128) :: version = '$Id: neural_convection.f90,v 1 2016/05/24 00:00:00 fms Exp $'
 character(len=128) :: tag = '$Name: fez $'

!-----------------------------------------------------------------------
!   ---- local/private data ----

    logical :: do_init=.true.

!-----------------------------------------------------------------------
!   ---- namelist ----

character(len=128) :: neural_filename = 'INPUT/neural_weights_v3.nc'

namelist /neural_convection_nml/ neural_filename

contains

!#######################################################################

   subroutine neural_convection (dt, tin, qin, pfull, phalf,  &
                                  rain, tdel, qdel,  &
                                   r_w1, r_w2, r_b1, r_b2, &
                               xscale_mean,xscale_stnd,yscale_absmax, conserve_energy_conv, &
                               raindebug, qdeldebug, &
                               dt0, dt1, dt2, dt3, dt4, dt5, dt6, dt7, dt8, dt9, &
                               dq0, dq1, dq2, dq3, dq4, dq5, dq6, dq7, dq8, dq9)

!-----------------------------------------------------------------------
!
!                     Betts-Miller Convection Scheme
!
!-----------------------------------------------------------------------
!
!   input:  dt       time step in seconds
!           tin      temperature at full model levels
!           qin      specific humidity of water vapor at full
!                      model levels
!           pfull    pressure at full model levels
!           phalf    pressure at half (interface) model levels
!
!  output:  rain     total precipitation (kg/m2)
!           tdel     temperature tendency at full model levels
!           qdel     specific humidity tendency (of water vapor) at
!                      full model levels
!
!-----------------------------------------------------------------------
!--------------------- interface arguments -----------------------------

   real   , intent(in) , dimension(:,:,:) :: tin, qin, pfull, phalf
   real   , intent(in)                    :: dt
   logical, intent(in)                    :: conserve_energy_conv
   real   , intent(out), dimension(:,:)   :: rain, raindebug
   real   , intent(out), dimension(:,:,:) :: tdel, qdel, qdeldebug, dt0, dt1, dt2, dt3
   real   , intent(out), dimension(:,:,:) :: dt4, dt5, dt6, dt7, dt8, dt9, dq0, dq1, dq2
   real   , intent(out), dimension(:,:,:) :: dq3, dq4, dq5, dq6, dq7, dq8, dq9
!-----------------------------------------------------------------------
!---------------------- local data -------------------------------------

   real,dimension(size(tin,1),size(tin,2))             :: precip, precipdebug
   real,dimension(16)                         :: qpc, tpc
   real,dimension(size(tin,3), 10)                         :: tdel_all, qdel_all
   real,dimension(32)                       :: features, targets
   real                                                :: deltak
 integer  i, j, k, ix, jx, kx, ktop, kx2, kx2ind, nx, n

      real, intent(in), dimension(32, 50, 10)   :: r_w1 
      real, intent(in), dimension(50, 10)       :: r_b1 
      real, intent(in), dimension(50, 32, 10)   :: r_w2
      real, intent(in), dimension(32, 10)        :: r_b2
      real, intent(in), dimension(32, 10)        :: xscale_mean
      real, intent(in), dimension(32, 10)        :: xscale_stnd
      real, intent(in), dimension(32, 10)        :: yscale_absmax

      real, dimension(50) :: z1, z2
!-----------------------------------------------------------------------
!     computation of precipitation by betts-miller scheme
!-----------------------------------------------------------------------
      if (do_init) call error_mesg ('neural_convection',  &
                         'neural_convection_init has not been called.', FATAL)
      ix=size(tin,1)
      jx=size(tin,2)
      kx=size(tin,3)
      kx2 = 16  ! Total number of levels the NN uses
      kx2ind = kx - kx2 + 1  ! Index value to grab data for NN from full profile
      nx = size(r_b1, 2) ! The number of NN ensemble members
       do i=1,ix
          do j=1,jx
             precip(i,j) = 0.
             precipdebug(i,j) = 0.
             tdel(i,j,:) = 0.
             qdel(i,j,:) = 0.
             dt0(i,j,:) = 0.
             dt1(i,j,:) = 0.
             dt2(i,j,:) = 0.
             dt3(i,j,:) = 0.
             dt4(i,j,:) = 0.
             dt5(i,j,:) = 0.
             dt6(i,j,:) = 0.
             dt7(i,j,:) = 0.
             dt8(i,j,:) = 0.
             dt9(i,j,:) = 0.
             dq0(i,j,:) = 0.
             dq1(i,j,:) = 0.
             dq2(i,j,:) = 0.
             dq3(i,j,:) = 0.
             dq4(i,j,:) = 0.
             dq5(i,j,:) = 0.
             dq6(i,j,:) = 0.
             dq7(i,j,:) = 0.
             dq8(i,j,:) = 0.
             dq9(i,j,:) = 0.
             tdel_all = 0.
             qdel_all = 0.
           do n=1,nx
! Initialize variables
             features = 0.
             qpc = 0.
             tpc = 0.
             z1 = 0.
             z2 = 0.
             targets = 0.
! Temperature and humidity profiles
             tpc = tin(i,j,kx2ind:kx)
             qpc = qin(i,j,kx2ind:kx)

! Combine tpc and rpc into a vector of levels (2*N_lev)
             features(1:kx2) = tpc
             features(kx2+1:2*kx2) = qpc
! Scale inputs
! See http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html#sklearn.preprocessing.MinMaxScaler.get_params for more info about scaling
             features = (features - xscale_mean(:,n)) / xscale_stnd(:,n)

! Apply trained regressor network to data using rectifier activation function
! forward prop to hidden layer
             z1 = matmul(features,  r_w1(:,:,n)) + r_b1(:,n)
! rectifier
             where (z1 .lt. 0.0)  z1 = 0.0
! forward prop to output layer
             targets = matmul(z1, r_w2(:,:,n)) + r_b2(:,n)
             targets = targets * yscale_absmax(:,n)
! Separate out targets into heating and moistening tendencies
             tdel_all(kx2ind:kx,n) = targets(1:kx2)
             qdel_all(kx2ind:kx,n) = targets(kx2 + 1:2*kx2)
           end do
! Correct units (nn trained on K/day and g/kg/day rather than K/s and kg/kg/s)
             tdel_all = tdel_all / 86400.  ! 3600. / 24.
             qdel_all = qdel_all / 86400000.  !3600. / 24. / 1000.
! Predicted temperature and humidity tendencies, but actually want the delta T and q
! So multiply each by the time step (now units of K and kg/kg)
             tdel_all = tdel_all * dt
             qdel_all = qdel_all * dt
! Now take mean over ensemble members
             tdel(i,j,:) = sum(tdel_all, 2) / 10.
             qdel(i,j,:) = sum(qdel_all, 2) / 10.
             dt0(i,j,:) = tdel_all(:,1)
             dt1(i,j,:) = tdel_all(:,2)
             dt2(i,j,:) = tdel_all(:,3)
             dt3(i,j,:) = tdel_all(:,4)
             dt4(i,j,:) = tdel_all(:,5)
             dt5(i,j,:) = tdel_all(:,6)
             dt6(i,j,:) = tdel_all(:,7)
             dt7(i,j,:) = tdel_all(:,8)
             dt8(i,j,:) = tdel_all(:,9)
             dt9(i,j,:) = tdel_all(:,10)
             dq0(i,j,:) = qdel_all(:,1)
             dq1(i,j,:) = qdel_all(:,2)
             dq2(i,j,:) = qdel_all(:,3)
             dq3(i,j,:) = qdel_all(:,4)
             dq4(i,j,:) = qdel_all(:,5)
             dq5(i,j,:) = qdel_all(:,6)
             dq6(i,j,:) = qdel_all(:,7)
             dq7(i,j,:) = qdel_all(:,8)
             dq8(i,j,:) = qdel_all(:,9)
             dq9(i,j,:) = qdel_all(:,10)
! If any humidities would become negative set them to zero (and recalc precip)
             do k=1, kx
                 if ( qin(i,j,k) + qdel(i,j,k) .lt. 0.0 ) then
                     ! Dry out level, but don't let it go negative
                     qdeldebug(i,j,k) = qdel(i,j,k)
                     qdel(i,j,k) = -qin(i,j,k)
                 endif
             end do

! Calculate precipitation
             do k=1, kx
                 precip(i,j) = precip(i,j) - qdel(i,j,k)*(phalf(i,j,k+1)- &
                            phalf(i,j,k))/grav
             end do
! If precipitation is negative, just set outputs to zero!
             if ( precip(i,j) .lt. 0.0 ) then 
                 precipdebug(i,j) = precip(i,j)
                 precip(i,j) = 0.0
                 tdel(i,j,:) = 0.0
                 qdel(i,j,:) = 0.0
             endif
! If we are conserving energy, do the below shift the temperature uniformly in the profile
             if (conserve_energy_conv) then
                deltak = 0.
                do k=kx2ind, kx
! Calculate the integrated difference in energy change within each level.
                   deltak = deltak - (tdel(i,j,k) + hlv/cp_air*&
                                     qdel(i,j,k))* &
                                     (phalf(i,j,k+1) - phalf(i,j,k))
                end do
! Divide by pressure up to level we do convection of
                deltak = deltak/(phalf(i,j,kx+1) - phalf(i,j,kx2ind))
! Adjust the reference profile (uniformly with height), and correspondingly
! the temperature change.
                tdel(i,j,kx2ind:kx) = tdel(i,j,kx2ind:kx) + deltak
             endif

          end do
       end do
       rain = precip
       raindebug = precipdebug
   end subroutine neural_convection





!#######################################################################


!##############################################################################
  subroutine check(status)

    ! checks error status after each netcdf, prints out text message each time
    !   an error code is returned. 

    integer, intent(in) :: status

    if(status /= nf90_noerr) then
       write(*, *) trim(nf90_strerror(status))
    end if
  end subroutine check

!#######################################################################


   subroutine neural_convection_init(r_w1, r_w2, r_b1, r_b2, &
                               xscale_mean,xscale_stnd, yscale_absmax)

!-----------------------------------------------------------------------
!
!        initialization for neural convection
!
!-----------------------------------------------------------------------
integer  unit,io,ierr
character(len=128) :: neural_filename = "neural_weights_v4.nc"

real, intent(out), dimension(32,50,10)    :: r_w1
real, intent(out), dimension(50,10)       :: r_b1
real, intent(out), dimension(50,32,10)    :: r_w2
real, intent(out), dimension(32,10)        :: r_b2
real, intent(out), dimension(32,10)        :: xscale_mean
real, intent(out), dimension(32,10)        :: xscale_stnd
real, intent(out), dimension(32,10)        :: yscale_absmax

! This will be the netCDF ID for the file and data variable.
integer :: ncid
integer :: r_w1_varid, r_b1_varid, r_w2_varid, r_b2_varid
integer :: xscale_mean_varid
integer :: xscale_stnd_varid, yscale_absmax_varid
!----------- read namelist ---------------------------------------------

      if (file_exist('input.nml')) then
         unit = open_file (file='input.nml', action='read')
         ierr=1; do while (ierr /= 0)
            read  (unit, nml=neural_convection_nml, iostat=io, end=10)
            ierr = check_nml_error (io,'neural_convection_nml')
         enddo
  10     call close_file (unit)
      endif

!---------- output namelist --------------------------------------------

      unit = open_file (file='logfile.out', action='append')
      if ( mpp_pe() == 0 ) then
           write (unit,'(/,80("="),/(a))') trim(version), trim(tag)
           write (unit,nml=neural_convection_nml)
      endif
      call close_file (unit)


      namelist /neural_convection_nml/ neural_filename

      write(*, *) 'Initializing neural weights'
      r_w1 = 0.
      r_w2 = 0.
      r_b1 = 0.
      r_b2 = 0.
      xscale_mean = 0.
      xscale_stnd = 0.
      yscale_absmax = 0.

! Open the file. NF90_NOWRITE tells netCDF we want read-only access
! Get the varid of the data variable, based on its name.
! Read the data.
      call check( nf90_open(     trim(neural_filename),NF90_SHARE,ncid ))

      call check( nf90_inq_varid(ncid,       "w1",        r_w1_varid))
      call check( nf90_get_var(  ncid,       r_w1_varid,    r_w1      ))
      
      call check( nf90_inq_varid(ncid,       "b1",        r_b1_varid))
      call check( nf90_get_var(  ncid,       r_b1_varid,    r_b1      ))

      call check( nf90_inq_varid(ncid,       "w2",        r_w2_varid))
      call check( nf90_get_var(  ncid,       r_w2_varid,    r_w2      ))
    
      call check( nf90_inq_varid(ncid,       "b2",        r_b2_varid))
      call check( nf90_get_var(  ncid,       r_b2_varid,    r_b2      ))
    
      call check( nf90_inq_varid(ncid,"xscale_mean",     xscale_mean_varid))
      call check( nf90_get_var(  ncid, xscale_mean_varid,xscale_mean      ))

      call check( nf90_inq_varid(ncid,"xscale_stnd",     xscale_stnd_varid))
      call check( nf90_get_var(  ncid, xscale_stnd_varid,xscale_stnd      ))
      
      call check( nf90_inq_varid(ncid,"yscale_absmax",     yscale_absmax_varid))
      call check( nf90_get_var(  ncid, yscale_absmax_varid,yscale_absmax      ))
    
      ! Close the file
      call check( nf90_close(ncid))

      write(*, *) 'Finished reading regression file.'

     do_init=.false.
   end subroutine neural_convection_init



!#######################################################################

end module neural_convection_mod

