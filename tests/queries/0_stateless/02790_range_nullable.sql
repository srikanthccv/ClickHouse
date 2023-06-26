SELECT range(toNullable(1));
SELECT range(0::Nullable(UInt64), 10::Nullable(UInt64), 2::Nullable(UInt64));
SELECT range(0::Nullable(Int64), 10::Nullable(Int64), 2::Nullable(Int64));
SELECT range(null); -- { serverError ILLEGAL_TYPE_OF_ARGUMENT }
SELECT range(Null::Nullable(UInt64), 10::Nullable(UInt64), 2::Nullable(UInt64)); -- { serverError BAD_ARGUMENTS }
SELECT range(0::Nullable(UInt64), Null::Nullable(UInt64), 2::Nullable(UInt64)); -- { serverError BAD_ARGUMENTS }
SELECT range(0::Nullable(UInt64), 10::Nullable(UInt64), Null::Nullable(UInt64)); -- { serverError BAD_ARGUMENTS }
