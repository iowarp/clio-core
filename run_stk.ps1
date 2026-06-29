$ErrorActionPreference = "Continue"
Set-Location "C:\Users\llogan\Documents\Projects\core"
& "C:\Program Files\CMake\bin\cmake.exe" --build build-boost --config Debug -j $env:NUMBER_OF_PROCESSORS *>&1 | Tee-Object -FilePath "stk.bld.log"
$be = $LASTEXITCODE
"BLD_EXIT=$be" | Set-Content "stk.bld.done"
if ($be -eq 0) {
  & "C:\Program Files\CMake\bin\ctest.exe" --test-dir build-boost --build-config Debug --output-on-failure --timeout 180 -LE ollama *>&1 | Tee-Object -FilePath "stk.test.log"
  "CTEST_EXIT=$LASTEXITCODE" | Set-Content "stk.test.done"
}
