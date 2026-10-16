// Mirrors firmware/esp32_wristband/include/ble_uuids.h and
// android/app/src/main/kotlin/com/sahara/app/BleUuids.kt. See docs/ble_protocol.md.
// The BLE link itself is owned by the native service so it survives the screen being off;
// these constants document the contract on the Dart side.

const String bandNamePrefix = 'SAHARA';
const String saharaServiceUuid = 'c7a10000-5a5a-4b3d-9a8e-5348415241a1';
const String alertCharacteristicUuid = 'c7a10001-5a5a-4b3d-9a8e-5348415241a1';
const String statusCharacteristicUuid = 'c7a10002-5a5a-4b3d-9a8e-5348415241a1';

const int bleProtocolVersion = 0x01;
const int cmdAlert = 0x01;
const int cmdTest = 0x02;
const int cmdStop = 0x03;
