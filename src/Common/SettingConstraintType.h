#pragma once


namespace DB
{

enum class SettingConstraintType
{
    // Default. Behave in the same way as WRITABLE, but is not inherited unless `settings_constraints_replace_previous` is set.
    NONE,

    // Setting can be change within specified range only in `readonly=0` or `readonly=2` mode.
    WRITABLE,

    // Setting cannot be changed at all.
    // Either READONLY or CONST keyword in SQL syntax can be used (<readonly/> or <const/> in config.xml) to enable this.
    // NOTE: name `CONST` is chosen to avoid confusion with `readonly` setting.
    CONST,

    // Setting can be changed within specified range, regardless of `readonly` setting value.
    CHANGEABLE_IN_READONLY,
};

}
