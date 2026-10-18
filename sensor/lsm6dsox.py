"""LSM6DSOX (ST, 6-DoF IMU: 3-axis accelerometer + 3-axis gyroscope + die temperature).

The Adafruit STEMMA QT breakout talks I2C, so a microcontroller reads it and streams lines
over USB serial (see lsm6dsox_serial/lsm6dsox_serial.ino). This module holds the datasheet
constants, decodes raw register bytes, and parses the serial lines.

Readings the part provides (ST DS13310):
  accelerometer   X/Y/Z, full scale ±2/±4/±8/±16 g, ODR 1.6 Hz - 6.66 kHz
  gyroscope       X/Y/Z, full scale ±125/±250/±500/±1000/±2000 dps, ODR 12.5 Hz - 6.66 kHz
  temperature     16-bit, 256 LSB/°C, 0 LSB = 25 °C, refreshed at 52 Hz
The temperature is the chip's own die temperature. It follows the board and the air around it,
not skin or core body temperature.
"""

import math
import struct

I2C_ADDR = 0x6A            # 0x6B with SA0 high (Adafruit default is 0x6A)
WHO_AM_I = 0x0F            # reads 0x6C
WHO_AM_I_VALUE = 0x6C
OUT_TEMP_L = 0x20          # temp L/H, then gyro X/Y/Z L/H (0x22), then accel X/Y/Z L/H (0x28)
OUTX_L_G = 0x22
OUTX_L_A = 0x28

ACC_MG_PER_LSB = {2: 0.061, 4: 0.122, 8: 0.244, 16: 0.488}
GYRO_MDPS_PER_LSB = {125: 4.375, 250: 8.75, 500: 17.5, 1000: 35.0, 2000: 70.0}
TEMP_LSB_PER_C = 256.0
TEMP_OFFSET_C = 25.0


def decode_block(raw: bytes, acc_fs_g: int = 4, gyro_fs_dps: int = 2000):
    """14 bytes read from OUT_TEMP_L -> (temp_c, (gx, gy, gz) dps, (ax, ay, az) mG)."""
    t, gx, gy, gz, ax, ay, az = struct.unpack("<7h", raw[:14])
    g = GYRO_MDPS_PER_LSB[gyro_fs_dps] / 1000
    a = ACC_MG_PER_LSB[acc_fs_g]
    return TEMP_OFFSET_C + t / TEMP_LSB_PER_C, (gx * g, gy * g, gz * g), (ax * a, ay * a, az * a)


def parse_line(line: str):
    """One serial line 'ax_mg,ay_mg,az_mg,gx_dps,gy_dps,gz_dps,temp_c' -> (acc, gyro, temp) or None.

    Lines starting with '#' (status from the board) and malformed lines return None.
    """
    line = line.strip()
    if not line or line.startswith("#"):
        return None
    try:
        v = [float(x) for x in line.split(",")]
    except ValueError:
        return None
    if len(v) != 7 or not all(math.isfinite(x) for x in v):
        return None
    return tuple(v[0:3]), tuple(v[3:6]), v[6]
