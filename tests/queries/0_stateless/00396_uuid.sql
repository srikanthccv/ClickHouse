SELECT UUIDNumToString(toFixedString(unhex('0123456789ABCDEF0123456789ABCDEF' AS hex) AS bytes, 16) AS uuid_binary) AS uuid_string, hex(UUIDStringToNum(uuid_string)) = hex AS test1, UUIDStringToNum(uuid_string) = bytes AS test2;
SELECT UUIDNumToString(toFixedString(unhex(materialize('0123456789ABCDEF0123456789ABCDEF') AS hex) AS bytes, 16) AS uuid_binary) AS uuid_string, hex(UUIDStringToNum(uuid_string)) = hex AS test1, UUIDStringToNum(uuid_string) = bytes AS test2;
SELECT hex(UUIDStringToNum('01234567-89ab-cdef-0123-456789abcdef'));
SELECT hex(UUIDStringToNum(materialize('01234567-89ab-cdef-0123-456789abcdef')));
SELECT '01234567-89ab-cdef-0123-456789abcdef' AS str, UUIDNumToString(UUIDStringToNum(str)), UUIDNumToString(UUIDStringToNum(toFixedString(str, 36)));
SELECT materialize('01234567-89ab-cdef-0123-456789abcdef') AS str, UUIDNumToString(UUIDStringToNum(str)), UUIDNumToString(UUIDStringToNum(toFixedString(str, 36)));
SELECT toString(toUUID('3f1ed72e-f7fe-4459-9cbe-95fe9298f845'));

-- conversion back and forth to big-endian hex string
with generateUUIDv4() as uuid,
    identity(lower(hex(reverse(reinterpretAsString(uuid))))) as str,
    reinterpretAsUUID(reverse(unhex(str))) as uuid2
select uuid = uuid2;

with generateUUIDv4() as uuid,
    hex(reverse(reinterpretAsString(uuid))) as str1,
    hex(uuid) as str2
select str1 = str2;

with generateUUIDv4() as uuid,
    bin(reverse(reinterpretAsString(uuid))) as bin_str1,
    bin(uuid) as bin_str2
select bin_str1 = bin_str2;