project "MLTest"
    location ""
    kind "ConsoleApp"
    language "C++"
    cppdialect "C++17"
    staticruntime "on"
    
    targetdir ("../bin/" .. MLOutputDir .. "/MLTest")
    objdir ("../bin-int/" .. MLOutputDir .. "/MLTest")
    
    files
    {
        "src/**.h",
        "src/**.cpp",
        "vendor/eigen/Eigen/**"
    }
    
    includedirs
    {
        "src",
        "vendor/eigen"
    }

    links
    {
    }

    filter "system:windows"
        systemversion "latest"

        defines
        {
            "ML_PLATFORM_WINDOWS",
            "_CRT_SECURE_NO_WARNINGS",
            "NOMINMAX"
        }

    filter "system:linux"
        systemversion "latest"

        defines
        {
            "ML_PLATFORM_LINUX",
        }

    filter "system:macosx"
        systemversion "latest"

    filter "configurations:Debug"
        defines "ML_DEBUG"
        runtime "Debug"
        symbols "on"

    filter "configurations:Release"
        defines "ML_RELEASE"
        runtime "Release"
        optimize "on"

    filter "configurations:Dist"
        defines "ML_DIST"
        runtime "Release"
        optimize "on"