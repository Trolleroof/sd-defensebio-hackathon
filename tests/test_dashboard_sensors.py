import time

from dashboard.server import Person, Team


def test_motion_uses_recent_samples_and_exposes_axes():
    person = Person("wearer", "Wearer", "Wearer", "LIVE", imu=True, connected=True)
    team = Team(None, (0, 0))
    team.add(person)
    person.add_imu((0, 0, 1000), (0, 0, 10), 26)
    person.add_imu((0, 0, 1100), (0, 0, -20), 26)
    row = team.tick()["people"][0]
    assert row["mv"] == 50
    assert row["rot"] == 15
    assert row["acc_xyz"] == [0, 0, 1100]
    assert row["gyro_xyz"] == [0, 0, -20]

    person.acc = type(person.acc)((t - 3, value) for t, value in person.acc)
    person.gyro = type(person.gyro)((t - 3, value) for t, value in person.gyro)
    person.imu_t = time.time() - 3
    row = team.tick()["people"][0]
    assert row["mv"] is None
    assert "rot" not in row
    assert "acc_xyz" not in row


def test_heart_rate_expires_while_imu_keeps_reporting():
    person = Person("wearer", "Wearer", "Wearer", "LIVE", imu=True, connected=True)
    team = Team(None, (0, 0))
    team.add(person)
    person.add_hr(80)
    person.hr_t -= 11
    person.add_imu((0, 0, 1000), (0, 0, 0), 26)
    assert team.tick()["people"][0]["hr"] is None
