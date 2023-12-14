{
  description = "simple python setup";

  inputs.flake-utils.url = "github:numtide/flake-utils";

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem
      (system:
        let pkgs = nixpkgs.legacyPackages.${system}; in
        {
          devShells.default = pkgs.mkShell {
                nativeBuildInputs = with pkgs; [
                    python311
                    python311Packages.numpy
                    python311Packages.matplotlib
                    python311Packages.pandas
                    python311Packages.streamlit
                    python311Packages.polars

                ];
            };
        }
      );
}
