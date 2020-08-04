workspace "MLTest"
    architecture "x64"

    configurations
    {
        "Debug",
        "Release",
        "Dist"
    }

    flags
    {
        "MultiProcessorCompile"
    }

    MLOutputDir = "%{cfg.buildcfg}-%{cfg.system}-%{cfg.architecture}"

    include ("MLTest")
