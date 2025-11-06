from dataclasses import dataclass
from typing import Optional, Sequence

from . import constants


@dataclass
class Material:
    """Constructs MuJoCo-compatible materials from MTL definitions."""

    name: str
    Ka: Optional[str] = None
    Kd: Optional[str] = None
    Ks: Optional[str] = None
    d: Optional[str] = None
    Tr: Optional[str] = None
    Ns: Optional[str] = None
    map_Kd: Optional[str] = None

    @staticmethod
    def from_string(lines: Sequence[str]) -> "Material":
        """Build a Material from lines of an MTL file."""
        attrs = {"name": lines[0].split(" ")[1].strip()}
        for line in lines[1:]:
            for attr in constants.MTL_FIELDS:
                if line.startswith(attr):
                    elems = line.split(" ")[1:]
                    elems = [elem for elem in elems if elem != ""]
                    attrs[attr] = " ".join(elems)
                    break
        return Material(**attrs)

    def mjcf_rgba(self) -> str:
        Kd = self.Kd or "1.0 1.0 1.0"
        if self.d is not None:
            alpha = self.d
        elif self.Tr is not None:
            alpha = str(1.0 - float(self.Tr))
        else:
            alpha = "1.0"
        return f"{Kd} {alpha}"

    def mjcf_shininess(self) -> str:
        if self.Ns is not None:
            Ns = float(self.Ns) / 1_000
        else:
            Ns = 0.5
        return f"{Ns}"

    def mjcf_specular(self) -> str:
        if self.Ks is not None:
            Ks = sum(list(map(float, self.Ks.split(" ")))) / 3
        else:
            Ks = 0.5
        return f"{Ks}"
