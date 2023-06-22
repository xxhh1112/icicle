      MOV R1, c[0x0][0x28] 
      S2R R8, SR_CTAID.X 
      ISETP.GE.U32.AND P0, PT, R8, c[0x0][0x17c], PT 
@P0   EXIT 
      S2R R7, SR_TID.X 
      ISETP.GE.U32.AND P0, PT, R7, c[0x0][0x0], PT 
@P0   EXIT 
      IMAD.MOV.U32 R0, RZ, RZ, c[0x0][0x0] 
      IMAD.MOV.U32 R24, RZ, RZ, c[0x0][0x180] 
      IMAD.SHL.U32 R0, R0, 0x2, RZ 
      I2F.U32.RP R4, R0 
      IADD3 R5, RZ, -R0, RZ 
      ISETP.NE.U32.AND P2, PT, R0, RZ, PT 
      MUFU.RCP R4, R4 
      IADD3 R2, R4, 0xffffffe, RZ 
      F2I.FTZ.U32.TRUNC.NTZ R3, R2 
      IMAD.MOV.U32 R2, RZ, RZ, RZ 
      IMAD R5, R5, R3, RZ 
      IMAD.HI.U32 R3, R3, R5, R2 
      IMAD.HI.U32 R5, R3, c[0x0][0x168], RZ 
      IMAD.MOV R3, RZ, RZ, -R5 
      IMAD R3, R0, R3, c[0x0][0x168] 
      ISETP.GE.U32.AND P0, PT, R3, R0, PT 
@P0   IADD3 R3, -R0, R3, RZ 
@P0   IADD3 R5, R5, 0x1, RZ 
      ISETP.GE.U32.AND P1, PT, R3, R0, PT 
@P1   IADD3 R5, R5, 0x1, RZ 
@!P2  LOP3.LUT R5, RZ, R0, RZ, 0x33, !PT 
      ISETP.GT.U32.AND P2, PT, R24, c[0x0][0x184], PT 
      I2F.U32.RP R4, R5 
      ISETP.NE.U32.AND P1, PT, R5, RZ, PT 
      MUFU.RCP R4, R4 
      IADD3 R2, R4, 0xffffffe, RZ 
      F2I.FTZ.U32.TRUNC.NTZ R3, R2 
      MOV R2, RZ 
      IMAD R6, R5, R3, RZ 
      IMAD.MOV R9, RZ, RZ, -R6 
      IMAD.HI.U32 R3, R3, R9, R2 
      IMAD.HI.U32 R3, R3, R8, RZ 
      IMAD.MOV R3, RZ, RZ, -R3 
      IMAD R6, R5, R3, R8 
      ISETP.GE.U32.AND P0, PT, R6, R5, PT 
@P0   IADD3 R6, -R5, R6, RZ 
      ISETP.GE.U32.AND P0, PT, R6, R5, PT 
@P0   IADD3 R6, -R5, R6, RZ 
@!P1  LOP3.LUT R6, RZ, R5, RZ, 0x33, !PT 
@P2   EXIT 
      IMAD R2, R6, c[0x0][0x0], R7 
      IADD3 R3, -R24, c[0x0][0x184], RZ 
      IMAD R0, R0, R8, RZ 
      MOV R25, c[0x0][0x180] 
      ULDC UR4, c[0x0][0x168] 
      IADD3 R24, R24, -c[0x0][0x184], RZ 
      UIADD3 UR4, UR4, 0xffff, URZ 
      LOP3.LUT R26, R2, 0xffff, RZ, 0xc0, !PT 
      ULDC.64 UR8, c[0x0][0x118] 
      IMAD.MOV.U32 R4, RZ, RZ, 0x1 
      SHF.R.U32.HI R5, RZ, R3, R26 
      ISETP.NE.AND P0, PT, R25, RZ, PT 
      SHF.L.U32 R4, R4, R3, RZ 
      LOP3.LUT R29, R4, 0xffff, RZ, 0xc0, !PT 
      IMAD R6, R29, R5, RZ 
      IADD3 R5, R4, -0x1, RZ 
      IMAD.SHL.U32 R6, R6, 0x2, RZ 
      LOP3.LUT R5, R5, R2, RZ, 0xc0, !PT 
      LOP3.LUT R6, R6, UR4, RZ, 0xc0, !PT 
      IMAD.IADD R5, R5, 0x1, R6 
      LOP3.LUT R5, R5, 0xffff, RZ, 0xc0, !PT 
      IMAD.IADD R7, R29, 0x1, R5 
@!P0  BRA 0x7f408ecfe5d0 
      UMOV UR5, 0x0 
      ISETP.NE.AND P0, PT, R24, RZ, PT 
      ULDC.64 UR6, c[0x0][0x18] 
      UIADD3 UR5, UP0, UR5, UR6, URZ 
      UIADD3.X UR6, URZ, UR7, URZ, UP0, !UPT 
      MOV R30, UR5 
      MOV R31, UR6 
      IMAD.WIDE.U32 R30, R5, 0x20, R30 
      IMAD.MOV.U32 R33, RZ, RZ, R31 
      MOV R32, R30 
      IMAD.WIDE.U32 R28, R29, 0x20, R30 
      IMAD.MOV.U32 R35, RZ, RZ, R29 
      MOV R34, R28 
@P0   BRA 0x7f408ecfe6d0 
      IADD3 R31, P0, R0, R5, RZ 
      IADD3 R29, P1, R0, R7, RZ 
      IMAD.X R6, RZ, RZ, RZ, P0 
      LEA R30, P0, R31, c[0x0][0x160], 0x5 
      IMAD.X R4, RZ, RZ, RZ, P1 
      LEA R28, P1, R29, c[0x0][0x160], 0x5 
      LEA.HI.X R31, R31, c[0x0][0x164], R6, 0x5, P0 
      LEA.HI.X R29, R29, c[0x0][0x164], R4, 0x5, P1 
      BRA 0x7f408ecfe6d0 
      UMOV UR5, 0x0 
      IADD3 R7, P1, R0, R7, RZ 
      ULDC.64 UR6, c[0x0][0x18] 
      IADD3 R33, P0, R0, R5, RZ 
      UIADD3 UR5, UP0, UR5, UR6, URZ 
      IMAD.X R4, RZ, RZ, RZ, P1 
      LEA R34, P1, R7, c[0x0][0x160], 0x5 
      UIADD3.X UR6, URZ, UR7, URZ, UP0, !UPT 
      IMAD.X R6, RZ, RZ, RZ, P0 
      LEA R32, P2, R33, c[0x0][0x160], 0x5 
      IMAD.U32 R30, RZ, RZ, UR5 
      LEA.HI.X R35, R7, c[0x0][0x164], R4, 0x5, P1 
      IMAD.U32 R31, RZ, RZ, UR6 
      LEA.HI.X R33, R33, c[0x0][0x164], R6, 0x5, P2 
      IMAD.WIDE.U32 R30, R5, 0x20, R30 
      IMAD.WIDE.U32 R28, R29, 0x20, R30 
      LD.E.128 R4, [R32.64] 
      LD.E.128 R8, [R34.64] 
      LD.E.128 R12, [R32.64+0x10] 
      LD.E.128 R16, [R34.64+0x10] 
      IADD3 R25, R25, 0x1, RZ 
      IADD3 R3, R3, -0x1, RZ 
      IADD3 R24, R24, 0x1, RZ 
      IADD3 R20, P0, R4, -R8, RZ 
      IADD3.X R21, P0, R5, ~R9, RZ, P0, !PT 
      IADD3.X R22, P1, R6, ~R10, RZ, P0, !PT 
      IADD3 R8, P0, R4, R8, RZ 
      IADD3.X R23, P1, R7, ~R11, RZ, P1, !PT 
      IADD3.X R9, P0, R5, R9, RZ, P0, !PT 
      IADD3.X R4, P1, R12, ~R16, RZ, P1, !PT 
      IADD3.X R10, P0, R6, R10, RZ, P0, !PT 
      IADD3.X R5, P2, R13, ~R17, RZ, P1, !PT 
      IADD3.X R11, P1, R7, R11, RZ, P0, !PT 
      IADD3 R27, P0, R8, -0x1, RZ 
      IADD3.X R6, P2, R14, ~R18, RZ, P2, !PT 
      IADD3.X R12, P1, R12, R16, RZ, P1, !PT 
      IADD3.X R16, P0, RZ, R9, RZ, P0, !PT 
      IADD3.X R7, P2, R15, ~R19, RZ, P2, !PT 
      IADD3.X R13, P1, R13, R17, RZ, P1, !PT 
      IADD3.X R33, P0, R10, 0x1a401, RZ, P0, !PT 
      IADD3.X R14, P1, R14, R18, RZ, P1, !PT 
      IADD3.X R34, P0, R11, -0x53bda403, RZ, P0, !PT 
      IMAD.X R15, R15, 0x1, R19, P1 
      IADD3.X R17, P1, R12, -0x9a1d806, RZ, P0, !PT 
@!P2  IADD3 R20, P0, R20, 0x1, RZ 
      IADD3.X R18, P1, R13, -0x3339d809, RZ, P1, !PT 
@!P2  IADD3.X R21, P0, R21, -0x1, RZ, P0, !PT 
      IADD3.X R19, P1, R14, -0x299d7d49, RZ, P1, !PT 
@!P2  IADD3.X R22, P0, R22, -0x1a402, RZ, P0, !PT 
      IADD3.X R32, P1, R15, -0x73eda754, RZ, P1, !PT 
@!P2  IADD3.X R23, P0, R23, 0x53bda402, RZ, P0, !PT 
      SEL R11, R34, R11, P1 
@!P2  IADD3.X R4, P0, R4, 0x9a1d805, RZ, P0, !PT 
      SEL R10, R33, R10, P1 
@!P2  IADD3.X R5, P0, R5, 0x3339d808, RZ, P0, !PT 
      SEL R9, R16, R9, P1 
@!P2  IADD3.X R6, P0, R6, 0x299d7d48, RZ, P0, !PT 
      SEL R8, R27, R8, P1 
@!P2  IADD3.X R7, R7, 0x73eda753, RZ, P0, !PT 
      ISETP.GT.U32.AND P0, PT, R25, c[0x0][0x184], PT 
      ST.E.128 [R30.64], R8 
      SEL R15, R32, R15, P1 
      SEL R14, R19, R14, P1 
      SEL R13, R18, R13, P1 
      SEL R12, R17, R12, P1 
      ST.E.128 [R30.64+0x10], R12 
      ST.E.128 [R28.64], R20 
      ST.E.128 [R28.64+0x10], R4 
      BAR.SYNC 0x0 
@!P0  BRA 0x7f408ecfe380 
      EXIT 
      BRA 0x7f408ecfea40
      NOP
      NOP
      NOP
      NOP
      NOP
      NOP
      NOP
      NOP
      NOP
      NOP
      NOP
