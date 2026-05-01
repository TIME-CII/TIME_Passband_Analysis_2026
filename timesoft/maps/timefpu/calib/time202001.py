from __future__ import division, print_function, unicode_literals, absolute_import

# TIME Cryostat with borrowed B3 SSAs and partial mK wiring

# Modified 2017-03-30

# Shunt resistor on the Nyquist chip
# Historically 3 mohms.  Recent B3 dip tests show 4 mohms is better.
# http://clk-1f.stanford.edu/twiki/bin/view/BICEP3/Logbook140930Dips_of_P4_shunts
R_SH = 0.004 # [Ohms] HF0 and HF1 use B3 nyquist chips.  Yes, HF1 uses B3 nyquists.

# ~ # BA NYQ13A wafer 5 and 6 were dipped by NIST and found to 3.0 mohm on average
# ~ R_SH = 0.003 # [Ohms] HF2/HF3 ?

# Total resistance of the wiring for a differential signal from the MCE
# backplane to the daughtercard.  Measured by probing the LN bias lines
# (ex. R between LN_BIAS_10 and LN_BIAS_RET_10) on the MCE backplane
# (remove a card and use the card extender).  LN bias lines are used
# since the shunt resistors are negligible.
# Measured for TIME cryostat with temporary wiring on 2017-03-22.
R_WIRES = 55 # Ohms

# TES bias resistance.  Sum of R52, R53, R64, and R65 on the bias cards.
# These four resistors are all in series with the LN bias lines.
# Verified with the TIME-Pilot bias card (Rev F6) on 2017-03-22.
R_TBIAS = 133+133+100+100 # Ohms

# Max output voltage of the LN bias lines (used for the TES biases).
# The DAC output can swing from -2.5V to 2.5V, but a negated version of
# the DAC output is output on the LN_BIAS_RET line, resulting in a
# differential output voltage ranging from -5V to 5V.
# Verified with the TIME-Pilot bias card (Rev F6) on 2017-03-22.
V_TBIAS_MAX = 2.5 * 2

# Number of bits in the LN bias DAC.  The DAC codes range from
# -(2^(n-1)) to +(2^(n-1) - 1), corresponding to voltages
# (DAC_CODE * V_TBIAS_MAX / 2^(N-1))
# Verified with the TIME-Pilot bias card (Rev F6) on 2017-03-22.
BITS_TBIAS = 16

# Multiplicative factor converting DAC code to TES bias current.
# Units of Amps / DAC_CODE
CAL_TES_BIAS_I = (V_TBIAS_MAX/(2**(BITS_TBIAS-1))) / (R_TBIAS + R_WIRES)

# Turn ratio between the readout inductor (not the nyquist chip inductor)
# and the SQ1 feedback inductor.
# Old Style: 15.5 for mux07a, 7.99 for mux09a
# New Style: 25 for mux11d (from Bryan 2017-03-30, need measured value)
M_FB1 = 25

# Full scale output of the SQ1 FB current DAC on the readout card.
# (1.2V reference, 2kohm FSR adjust resistor R36, 32x multiplier from
# reference current to output current)
# Verified resistor on TIME-Pilot readout card (Rev E1) on 2017-03-22.
I_FB1_MAX_DAC = 1.2/2000*32 # Amps

# Resistor R33 on the readout card that is in parallel with the SQ1 FB
# loop.  Since the other branch of the current divider is 2000+ Ohms,
# this resistor forms a constant voltage source.
# Verified resistor on TIME-Pilot readout card (Rev E1) on 2017-03-22.
R_FB1_PARALLEL = 50 # Ohms

# Max output voltage on the SQ1 FB line.  This is a unipolar output
# that is formed with a current DAC and a resistor.
# Verified ~0.96V on TIME-Pilot readout card (Rev E1) on 2017-03-22.
V_FB1_MAX = I_FB1_MAX_DAC * R_FB1_PARALLEL

# Number of bits in SQ1 FB current DAC.  Note that the output is shifted
# such that -8192 is 0V, 0 is mid scale, and 8191 is full scale.  However,
# since we can only measure TES branch current up to an unknown constant offset,
# we have no reason to correct for this.
# Verified part number on TIME-Pilot readout card (Rev E1) on 2017-03-22.
BITS_FB1 = 14

# Resistor R28 on the readout card.  In series with the SQ1 FB loop.
# Verified resistor on TIME-Pilot readout card (Rev E1) on 2017-03-22.
R_FB1_SERIES_RC = 50 # Ohms

# Series resistor on the MCE backplane (ex: R138).
# Verified resistor on TIME-Pilot mce backplane on 2017-03-22.
R_FB1_SERIES_BACKPLANE = 2000 # Ohms

# Total resistance in series with the SQ1 FB inductor
R_FB1_SERIES_TOTAL = R_FB1_SERIES_RC + R_FB1_SERIES_BACKPLANE + R_WIRES

# Multiplicative factor converting DAC code to TES current (the portion
# of the TES bias current that doesn't go through the shunt resistor).
# Includes the turn ratio between the SQ1 FB inductor and the TES
# readout inductor.
# Note that the resulting TES branch current will only be correct up
# to a constant, we cannot measure the absolute current because the
# squid is periodic.  Additionally, 18 bits of the raw SQ1 FB code are
# kept, low-pass-filtered, and sent to the user as data.  However, the
# actual DAC is only 14 bits (and there is a shift such that DAC code
# 0 is midscale output).  This is ok because of flux jumping, we can
# have meaningful values of the SQ1 FB code outside of the 14 bit range
# of the DAC.  The difference between two DAC units in Amps is well
# defined and is what we calibrate here.  Using unfilter='DC' and
# the default rescale=True when reading data ensures that the width of
# one DAC code in the resulting data matches the width of one physical
# DAC code.  Thus, despite having data that goes negative and stretches
# outside 2**BITS_FB1, this is the correct conversion factor.
# Units of Amps / DAC_CODE
CAL_SQ1_FB_I = (V_FB1_MAX/(2**BITS_FB1)) / R_FB1_SERIES_TOTAL / M_FB1

# Calibration tests (measuring voltage over test resistor)
# Does not test M_FB or R_SH.  Done by connecting a 100 pin MDM
# test board into the MCE (no cryo wiring) and monitoring the voltage
# over the 50 ohm test resistors.
# Verified 2017-03-22
TEST_DACCODE_TES = 8191
TEST_DACCODE_FB1 = 8191
TEST_R = 50 # Ohms
TEST_RESULT_TES = 0.112 # V
TEST_RESULT_FB1 = 0.0111 # V
CAL_TES_BIAS_I_EFFECTIVE = CAL_TES_BIAS_I * (R_TBIAS + R_WIRES) / (R_TBIAS + TEST_R)
CAL_SQ1_FB_I_EFFECTIVE = CAL_SQ1_FB_I * M_FB1 * R_FB1_SERIES_TOTAL / (R_FB1_SERIES_TOTAL - R_WIRES + TEST_R)
print("Calib Test Results (should be ~1):", round(CAL_TES_BIAS_I_EFFECTIVE*TEST_DACCODE_TES*TEST_R/TEST_RESULT_TES,2), round(CAL_SQ1_FB_I_EFFECTIVE*TEST_DACCODE_FB1*TEST_R/TEST_RESULT_FB1,2))

print("Calib Values:", CAL_TES_BIAS_I, CAL_SQ1_FB_I)

L_NYQ = 2.0e-6 # H (Keck 1.35 uH, SPIDER/B3/BA/TIME 2.0 uH)

R_REF_LOW = 0.95 # Ohms (1 ohm reference resistor, measured stable to <5% from 300 K to 260 mK with LS 370 2016-02-08)
R_REF_HIGH = 100 # Ohms (100 ohm cryogenic resistor, not measured here but assumed correct)

# Load curve measurement points
IMIN_BIAS_SC = 10e-6 # A
IMAX_BIAS_SC = 100e-6 # A
IMIN_BIAS_TI_RN = 500e-6 # A
IMAX_BIAS_TI_RN = 580e-6 # A
R_PSAT = 0.065 # Ohms
