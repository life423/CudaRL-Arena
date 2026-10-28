#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <filesystem>
#include <string>
int WINAPI wWinMain(HINSTANCE,HINSTANCE,PWSTR arguments,int) {
    wchar_t executable[32768]{};const DWORD length=GetModuleFileNameW(nullptr,executable,32768);
    if(length==0 || length>=32768) return 1;
    const auto root=std::filesystem::path(executable).parent_path();const auto engine=root/L"engine.exe",project=root/L"player";
    if(!std::filesystem::exists(engine) || !std::filesystem::exists(project/L"project.godot")) {
        MessageBoxW(nullptr,L"Keep CudaRLPond.exe, engine.exe and the player folder together. Extract the complete ZIP before opening the pond.",L"CudaRL Pond",MB_OK|MB_ICONERROR);return 1;
    }
    std::wstring command=L"\""+engine.wstring()+L"\" --path \""+project.wstring()+L"\" -- "+arguments;
    STARTUPINFOW startup{};startup.cb=sizeof(startup);PROCESS_INFORMATION process{};
    if(!CreateProcessW(engine.c_str(),command.data(),nullptr,nullptr,FALSE,0,nullptr,root.c_str(),&startup,&process)) {
        MessageBoxW(nullptr,L"The pond could not start. Check that all files were extracted and your graphics drivers are available.",L"CudaRL Pond",MB_OK|MB_ICONERROR);return 1;
    }
    CloseHandle(process.hThread);WaitForSingleObject(process.hProcess,INFINITE);DWORD result=1;GetExitCodeProcess(process.hProcess,&result);CloseHandle(process.hProcess);return static_cast<int>(result);
}
