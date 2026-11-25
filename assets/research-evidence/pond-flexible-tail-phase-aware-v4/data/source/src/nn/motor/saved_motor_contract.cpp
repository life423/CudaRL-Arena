#include "motor/saved_motor_contract.h"

namespace cudarl::motor {
const std::array<std::string_view,12>& saved_motor_source_contract_hashes() {
    // Original source order and LF-normalized SHA256 values are recorded in
    // assets/saved-motor-provenance.json. Never regenerate from changed code to
    // make an old package appear compatible. New runtime evidence is separate.
    static constexpr std::array<std::string_view,12> hashes={
        "5ab30eb92a609ac8cfd50944bef8898cf199be3989dbb6550b05bf44bc5449d5",
        "dc53d22cff99352c5e378b6bcd919569c9a26455aa5df9f8ec1c36172171314f",
        "994f7719f497e237ad4428c74feac2d73ce57e2ea6ea8e12ab8085882e28df22",
        "6d46b600194ce1c830fce11c01860143bbe3af9452996dc6bff07043f6c10332",
        "b6141e4af555edf1dcd826047a081016296883ee005c4a6abc7438a9d83cab4c",
        "9c6a9d3c31b09f8207c4091252c4f7ac963ff2b09b2b8ad3d278570b4f536e74",
        "fa437b0429cb9fb3f7233dfb00548be57eda74417f98d27fa4c28c31e69ee2bc",
        "29f79e704cdac48f7045734c50549afc4052cef8ed96715f8f57f49402f832a6",
        "3e88a73390f08803eb5f55abc8ab1d93533884169817bf5165f7da698dc152a6",
        "6fcab032bc114416ad5fc481da5f15483e7594201b8cbc343df888af7f1a4cd3",
        "f848a64e32c0bbefdf418972d42e82eab7214195f574c0638e1cc85206eed9cb",
        "086a3b10b170094c4d6eee5ee1175baec7cbd3f85670dc39207a3dd41b0ef7eb"
    };
    return hashes;
}
}
