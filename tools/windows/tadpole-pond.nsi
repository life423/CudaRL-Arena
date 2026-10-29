Unicode true
!include "MUI2.nsh"
!include "FileFunc.nsh"
!include "LogicLib.nsh"
!include "x64.nsh"
!ifndef PAYLOAD
  !error "PAYLOAD must name the verified Windows app folder"
!endif
!ifndef OUTPUT
  !error "OUTPUT must name the installer executable"
!endif
!ifndef APPICON
  !error "APPICON must name the application icon"
!endif
Name "Tadpole Pond"
OutFile "${OUTPUT}"
InstallDir "$LOCALAPPDATA\Programs\Tadpole Pond"
RequestExecutionLevel user
ManifestDPIAware true
SetCompressor /SOLID lzma
SetCompressorDictSize 32
AutoCloseWindow true
ShowInstDetails nevershow
ShowUninstDetails nevershow
BrandingText "Tadpole Pond · local AI learning"
VIProductVersion "0.1.0.2"
VIAddVersionKey "ProductName" "Tadpole Pond"
VIAddVersionKey "FileDescription" "Tadpole Pond installer"
VIAddVersionKey "FileVersion" "0.1.0-preview.2"
VIAddVersionKey "ProductVersion" "0.1.0-preview.2"
VIAddVersionKey "LegalCopyright" "CudaRL Arena project"
!define MUI_ICON "${APPICON}"
!define MUI_UNICON "${APPICON}"
!define MUI_INSTFILESPAGE_HEADER_TEXT "Getting your pond ready"
!define MUI_INSTFILESPAGE_HEADER_SUBTEXT "Installing for your account. The game will open automatically."
!insertmacro MUI_PAGE_INSTFILES
!insertmacro MUI_UNPAGE_CONFIRM
!insertmacro MUI_UNPAGE_INSTFILES
!insertmacro MUI_LANGUAGE "English"
Var TestMode
Var Parameters

Function .onInit
  SetShellVarContext current
  SetRegView 64
  StrCpy $TestMode 0
  ${GetParameters} $Parameters
  ClearErrors
  ${GetOptions} $Parameters "/TESTINSTALL" $0
  ${IfNot} ${Errors}
    StrCpy $TestMode 1
  ${EndIf}
  ${IfNot} ${RunningX64}
    MessageBox MB_OK "Tadpole Pond needs 64-bit Windows."
    Quit
  ${EndIf}
  ; Refuse to replace an app that's currently open. Never terminate its learner.
  IfFileExists "$INSTDIR\TadpolePond.exe" 0 ready
  ClearErrors
  FileOpen $0 "$INSTDIR\TadpolePond.exe" a
  IfErrors in_use
  FileClose $0
  Goto ready
  in_use:
    MessageBox MB_OK "Close Tadpole Pond, then open this installer again. Your saved progress will be kept."
    Quit
  ready:
FunctionEnd

Section "Tadpole Pond"
  SetOutPath "$INSTDIR"
  File /r "${PAYLOAD}\*.*"
  File /oname=TadpolePond.ico "${APPICON}"
  WriteUninstaller "$INSTDIR\Uninstall.exe"
  ${If} $TestMode == 0
    CreateShortcut "$DESKTOP\Tadpole Pond.lnk" "$INSTDIR\TadpolePond.exe" "" "$INSTDIR\TadpolePond.ico"
    CreateShortcut "$SMPROGRAMS\Tadpole Pond.lnk" "$INSTDIR\TadpolePond.exe" "" "$INSTDIR\TadpolePond.ico"
    WriteRegStr HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\TadpolePond" "DisplayName" "Tadpole Pond"
    WriteRegStr HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\TadpolePond" "DisplayVersion" "0.1.0-preview.2"
    WriteRegStr HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\TadpolePond" "Publisher" "CudaRL"
    WriteRegStr HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\TadpolePond" "InstallLocation" "$INSTDIR"
    WriteRegStr HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\TadpolePond" "DisplayIcon" "$INSTDIR\TadpolePond.ico"
    WriteRegStr HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\TadpolePond" "UninstallString" '$\"$INSTDIR\Uninstall.exe$\"'
    WriteRegStr HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\TadpolePond" "QuietUninstallString" '$\"$INSTDIR\Uninstall.exe$\" /S'
    WriteRegDWORD HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\TadpolePond" "NoModify" 1
    WriteRegDWORD HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\TadpolePond" "NoRepair" 1
  ${EndIf}
SectionEnd

Function .onInstSuccess
  ${If} $TestMode == 0
    IfSilent done
    Exec '$\"$INSTDIR\TadpolePond.exe$\"'
  ${EndIf}
  done:
FunctionEnd

Function un.onInit
  SetShellVarContext current
  SetRegView 64
  StrCpy $TestMode 0
  ${GetParameters} $Parameters
  ClearErrors
  ${GetOptions} $Parameters "/TESTINSTALL" $0
  ${IfNot} ${Errors}
    StrCpy $TestMode 1
  ${EndIf}
  IfFileExists "$INSTDIR\TadpolePond.exe" 0 ready
  ClearErrors
  FileOpen $0 "$INSTDIR\TadpolePond.exe" a
  IfErrors in_use
  FileClose $0
  Goto ready
  in_use:
    MessageBox MB_OK "Close Tadpole Pond before uninstalling. Your saved training progress will be kept."
    Quit
  ready:
FunctionEnd

Section "Uninstall"
  ; Remove only this app's known files. Leave saved progress and unrelated files.
  Delete "$INSTDIR\TadpolePond.exe"
  Delete "$INSTDIR\TadpolePond.pck"
  Delete "$INSTDIR\cudarl.dll"
  Delete "$INSTDIR\TadpolePond.ico"
  Delete "$INSTDIR\START-HERE.txt"
  Delete "$INSTDIR\release-manifest.json"
  Delete "$INSTDIR\data\selected-policy.bin"
  Delete "$INSTDIR\data\policies\deep_selector_recovery.bin"
  Delete "$INSTDIR\data\policies\deep_selector.bin"
  Delete "$INSTDIR\data\policies\shuffle_reference.json"
  Delete "$INSTDIR\data\policies\tadpole.json"
  Delete "$INSTDIR\licenses\cuda\LICENSE.txt"
  Delete "$INSTDIR\licenses\godot\LICENSE.txt"
  Delete "$INSTDIR\licenses\godot\COPYRIGHT.txt"
  Delete "$INSTDIR\licenses\godot\GODOT-CPP-LICENSE.md"
  Delete "$INSTDIR\licenses\nsis\LICENSE.txt"
  Delete "$INSTDIR\Uninstall.exe"
  RMDir "$INSTDIR\data\policies"
  RMDir "$INSTDIR\data"
  RMDir "$INSTDIR\licenses\cuda"
  RMDir "$INSTDIR\licenses\godot"
  RMDir "$INSTDIR\licenses\nsis"
  RMDir "$INSTDIR\licenses"
  ${If} $TestMode == 0
    Delete "$DESKTOP\Tadpole Pond.lnk"
    Delete "$SMPROGRAMS\Tadpole Pond.lnk"
    DeleteRegKey HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\TadpolePond"
  ${EndIf}
  RMDir "$INSTDIR"
SectionEnd
