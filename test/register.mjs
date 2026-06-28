// Register the extensionless-".ts" resolver hook for `node --import`.
import { register } from "node:module";
register("./resolve-ts.mjs", import.meta.url);
