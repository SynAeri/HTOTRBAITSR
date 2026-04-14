# Nix shell providing system libraries required by the Python venv (torch, numpy, opencv)
{ pkgs ? import <nixpkgs> {} }:

pkgs.mkShell {
  name = "htotrbaitsr";

  buildInputs = with pkgs; [
    python313
    stdenv.cc.cc.lib
    zlib
    libGL
    glib
  ];

  shellHook = ''
    export LD_LIBRARY_PATH=${pkgs.stdenv.cc.cc.lib}/lib:${pkgs.zlib}/lib:${pkgs.libGL}/lib:${pkgs.glib}/lib:$LD_LIBRARY_PATH
    source .venv/bin/activate
    echo "venv activated — python: $(which python3)"
  '';
}
