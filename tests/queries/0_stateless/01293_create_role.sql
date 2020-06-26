DROP ROLE IF EXISTS r1_01293, r2_01293, r3_01293, r4_01293, r5_01293, r6_01293, r7_01293, r8_01293, r9_01293;
DROP ROLE IF EXISTS r2_01293_renamed;
DROP ROLE IF EXISTS r1_01293@'%', 'r2_01293@%.myhost.com';

SELECT '-- default';
CREATE ROLE r1_01293;
SHOW CREATE ROLE r1_01293;

SELECT '-- same as default';
CREATE ROLE r2_01293 SETTINGS NONE;
SHOW CREATE ROLE r2_01293;

SELECT '-- rename';
ALTER ROLE r2_01293 RENAME TO 'r2_01293_renamed';
SHOW CREATE ROLE r2_01293; -- { serverError 511 } -- Role not found
SHOW CREATE ROLE r2_01293_renamed;
DROP ROLE r1_01293, r2_01293_renamed;

SELECT '-- host after @';
CREATE ROLE r1_01293@'%';
CREATE ROLE r2_01293@'%.myhost.com';
SHOW CREATE ROLE r1_01293@'%';
SHOW CREATE ROLE r1_01293;
SHOW CREATE ROLE r2_01293@'%.myhost.com';
SHOW CREATE ROLE 'r2_01293@%.myhost.com';
DROP ROLE r1_01293@'%', 'r2_01293@%.myhost.com';

SELECT '-- settings';
CREATE ROLE r1_01293 SETTINGS NONE;
CREATE ROLE r2_01293 SETTINGS PROFILE 'default';
CREATE ROLE r3_01293 SETTINGS max_memory_usage=5000000;
CREATE ROLE r4_01293 SETTINGS max_memory_usage MIN=5000000;
CREATE ROLE r5_01293 SETTINGS max_memory_usage MAX=5000000;
CREATE ROLE r6_01293 SETTINGS max_memory_usage READONLY;
CREATE ROLE r7_01293 SETTINGS max_memory_usage WRITABLE;
CREATE ROLE r8_01293 SETTINGS max_memory_usage=5000000 MIN 4000000 MAX 6000000 READONLY;
CREATE ROLE r9_01293 SETTINGS PROFILE 'default', max_memory_usage=5000000 WRITABLE;
SHOW CREATE ROLE r1_01293;
SHOW CREATE ROLE r2_01293;
SHOW CREATE ROLE r3_01293;
SHOW CREATE ROLE r4_01293;
SHOW CREATE ROLE r5_01293;
SHOW CREATE ROLE r6_01293;
SHOW CREATE ROLE r7_01293;
SHOW CREATE ROLE r8_01293;
SHOW CREATE ROLE r9_01293;
ALTER ROLE r1_01293 SETTINGS readonly=1;
ALTER ROLE r2_01293 SETTINGS readonly=1;
ALTER ROLE r3_01293 SETTINGS NONE;
SHOW CREATE ROLE r1_01293;
SHOW CREATE ROLE r2_01293;
SHOW CREATE ROLE r3_01293;
DROP ROLE r1_01293, r2_01293, r3_01293, r4_01293, r5_01293, r6_01293, r7_01293, r8_01293, r9_01293;

SELECT '-- multiple roles in one command';
CREATE ROLE r1_01293, r2_01293;
SHOW CREATE ROLE r1_01293, r2_01293;
ALTER ROLE r1_01293, r2_01293 SETTINGS readonly=1;
SHOW CREATE ROLE r1_01293, r2_01293;
DROP ROLE r1_01293, r2_01293;

SELECT '-- system.roles';
CREATE ROLE r1_01293;
SELECT name, storage from system.roles WHERE name='r1_01293';
DROP ROLE r1_01293;

SELECT '-- system.settings_profile_elements';
CREATE ROLE r1_01293 SETTINGS readonly=1;
CREATE ROLE r2_01293 SETTINGS PROFILE 'default';
CREATE ROLE r3_01293 SETTINGS max_memory_usage=5000000 MIN 4000000 MAX 6000000 WRITABLE;
CREATE ROLE r4_01293 SETTINGS PROFILE 'default', max_memory_usage=5000000, readonly=1;
CREATE ROLE r5_01293 SETTINGS NONE;
SELECT * FROM system.settings_profile_elements WHERE role_name LIKE 'r%\_01293' ORDER BY role_name, index;
DROP ROLE r1_01293, r2_01293, r3_01293, r4_01293, r5_01293;
