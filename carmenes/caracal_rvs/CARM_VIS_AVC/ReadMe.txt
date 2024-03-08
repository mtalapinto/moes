Format Description

default format (*.dat): columns separated by blanks time series

File Summary:
--------------------------------------------------------------------------------
FileName                 Explanations
--------------------------------------------------------------------------------
gtoc_arm_night_zero.txt  nightly zero-point (NZP) velocity (arm = vis or nir)
obj.avc.dat              Radial velocity (drift and nzp corrected, nan-drifts removed, and nan-NZP replaced by median NZP)
obj.avcn.dat             Radial velocity (drift and nzp corrected, nan-drifts treated as zero, and nan-NZP replaced by median NZP)
--------------------------------------------------------------------------------
NOTES:
[1]   (BJD,BERV) depending on serval input option, propagated from CARACAL
      or recomputed by SERVAL


Description of file: gtoc_arm_night_zero.txt
--------------------------------------------------------------------------------
Column Format Units     Label     Explanations
--------------------------------------------------------------------------------
     1 D      ---       BJD       Julian date at mid-day before the corrected night
     2 D      m/s       NZP       Nightly zero point (NZP)
     3 D      m/s     E_NZP       Error for NZP
     4 D      ---       N_RV      Number of RV-quiet star RVs used to calculate the NZP
     5 I      ---       FLAG_NZP  Byte flag: 0 (ok), 1 (NZP replaced by median NZP)
--------------------------------------------------------------------------------

Description of file: obj.avc.dat
--------------------------------------------------------------------------------
Column Format Units     Label     Explanations
--------------------------------------------------------------------------------
     1 D      ---       BJD       Barycentric Julian date [1]
     2 D      m/s       AVC       Radial velocity (drift and NZP corrected)
     3 D      m/s     E_AVC       Radial velocity error
--------------------------------------------------------------------------------


Description of file: obj.avcn.dat
--------------------------------------------------------------------------------
Column Format Units     Label     Explanations
--------------------------------------------------------------------------------
     1 D      ---       BJD       Barycentric Julian date [1]
     2 D      m/s       AVC       Radial velocity (drift and NZP corrected)
     3 D      m/s     E_AVC       Radial velocity error
     4 D      m/s       DRIFT     CARACAL drift measure
     5 D      m/s     E_DRIFT     CARACAL drift measure error
     6 D      m/s       RV        Radial velocity
     7 D      m/s     E_RV        Radial velocity error
     8 D      km/s      BERV      Barycentric earth radial velocity [1]
     9 D      m/s       SADRIFT   Drift due to secular acceleration
    10 D      m/s       NZP       Nightly zero point (NZP)
    11 D      m/s     E_NZP       Error for NZP
    12 I      ---       FLAG_AVC  Byte flag: 0 (ok), 1 (NZP replaced by median NZP), 2 (nan drift)
--------------------------------------------------------------------------------
