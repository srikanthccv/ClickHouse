select * from format(BSONEachRow, 'x UInt32, y UInt32', x'1a0000001078002a0000001078002a0000001079002a00000000'); -- {serverError INCORRECT_DATA}
