SELECT INET6_NTOA(toFixedString(unhex('2A0206B8000000000000000000000011'), 16));
SELECT hex(INET6_ATON('2a02:6b8::11'));
SELECT INET_NTOA(toUInt32(1337));
SELECT INET_ATON('192.168.0.1');
