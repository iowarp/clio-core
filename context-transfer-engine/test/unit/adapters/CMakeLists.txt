# Adapter Unit Tests

# Add subdirectories for each adapter based on enabled options
# Note: All adapters require WRP_CORE_ENABLE_ELF=ON except ADIOS2
if(WRP_CORE_ENABLE_ELF AND WRP_CTE_ENABLE_POSIX_ADAPTER)
    add_subdirectory(posix)
endif()

# ADIOS2 adapter tests (doesn't require ELF interception)
# Only add if ADIOS2 adapter is enabled
if(WRP_CTE_ENABLE_ADIOS2_ADAPTER)
    add_subdirectory(adios2)
endif()