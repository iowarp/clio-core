/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved. BSD 3-Clause license.
 */

/**
 * @file test_chunk_lineage.cc
 * @brief Blob name -> (lineage, timestep): the identity that lets the SAME
 * logical block be recognised across simulation timesteps.
 *
 * The four drivers in this benchmark name blobs two different ways --
 * timestep-first (`step00010/E_x/chunk_0`, the field replay) and
 * timestep-second (`force/step_100/chunk_2`, the in-situ drivers) -- so the
 * parser is written against BOTH, and every case below is a real name taken
 * from an explore.csv rather than an invented one.
 *
 * MainPretest()/MainPosttest() are defined once per binary in test_models.cc.
 */
#include <string>
#include <vector>

#include "basic_test.h"
#include "clio_ctp/compress/preprocess/chunk_lineage.h"

namespace {
using ctp::compress::preprocess::BlobLineage;
using ctp::compress::preprocess::ParseBlobLineage;
}  // namespace

/* Same logical block at two timesteps must land on ONE key -- the whole
   feature rests on this, in both naming conventions. */
TEST_CASE("PredictionReuseLineageSameBlockAcrossTimesteps") {
  // Field replay: timestep is the FIRST component.
  const BlobLineage a = ParseBlobLineage("step00000/E_x/chunk_0");
  const BlobLineage b = ParseBlobLineage("step00010/E_x/chunk_0");
  REQUIRE(a.resolved);
  REQUIRE(b.resolved);
  REQUIRE(a.key == b.key);
  REQUIRE(a.key == "E_x/chunk_0");
  REQUIRE(a.timestep == 0);
  REQUIRE(b.timestep == 10);

  // In-situ: timestep is the SECOND component.
  const BlobLineage c = ParseBlobLineage("force/step_0/chunk_2");
  const BlobLineage d = ParseBlobLineage("force/step_100/chunk_2");
  REQUIRE(c.resolved);
  REQUIRE(d.resolved);
  REQUIRE(c.key == d.key);
  REQUIRE(c.key == "force/chunk_2");
  REQUIRE(c.timestep == 0);
  REQUIRE(d.timestep == 100);

  // Nyx's AMReX fabs: `plt` prefix, and a FIELD name that itself contains
  // digits (`fab0000_comp00_density`) -- which must NOT be mistaken for the
  // timestep component.
  const BlobLineage e = ParseBlobLineage("plt00000/fab0000_comp00_density/chunk_0");
  const BlobLineage f = ParseBlobLineage("plt00007/fab0000_comp00_density/chunk_0");
  REQUIRE(e.resolved);
  REQUIRE(e.key == f.key);
  REQUIRE(e.key == "fab0000_comp00_density/chunk_0");
  REQUIRE(e.timestep == 0);
  REQUIRE(f.timestep == 7);

  // VPIC.
  const BlobLineage g = ParseBlobLineage("cbx/step_00025/chunk_0");
  const BlobLineage h = ParseBlobLineage("cbx/step_00050/chunk_0");
  REQUIRE(g.key == h.key);
  REQUIRE(g.key == "cbx/chunk_0");
  REQUIRE(g.timestep == 25);
  REQUIRE(h.timestep == 50);
}

/* Different blocks must not collide -- a collision differences two unrelated
   fields against each other and reuses one's prediction for the other. */
TEST_CASE("PredictionReuseLineageDistinctBlocksDoNotCollide") {
  // Different chunk of the same field.
  REQUIRE(ParseBlobLineage("step00000/E_x/chunk_0").key !=
          ParseBlobLineage("step00000/E_x/chunk_1").key);
  // Same chunk index, DIFFERENT variable.
  REQUIRE(ParseBlobLineage("step00000/E_x/chunk_5").key !=
          ParseBlobLineage("step00000/E_y/chunk_5").key);
  // Same chunk index, different variable, in-situ convention.
  REQUIRE(ParseBlobLineage("force/step_0/chunk_5").key !=
          ParseBlobLineage("velocity/step_0/chunk_5").key);
  // A field whose name differs only in a digit.
  REQUIRE(ParseBlobLineage("plt00000/fab0000_comp00_density/chunk_0").key !=
          ParseBlobLineage("plt00000/fab0000_comp01_xmom/chunk_0").key);
}

/* Arrival ORDER must not affect identity: the key is a pure function of the
   name, so blocks reordered between timesteps still pair up. */
TEST_CASE("PredictionReuseLineageIsOrderIndependent") {
  const std::vector<std::string> t1 = {"step00000/A/chunk_0",
                                       "step00000/B/chunk_0",
                                       "step00000/C/chunk_0"};
  const std::vector<std::string> t2 = {"step00010/C/chunk_0",
                                       "step00010/A/chunk_0",
                                       "step00010/B/chunk_0"};
  REQUIRE(ParseBlobLineage(t1[0]).key == ParseBlobLineage(t2[1]).key);
  REQUIRE(ParseBlobLineage(t1[1]).key == ParseBlobLineage(t2[2]).key);
  REQUIRE(ParseBlobLineage(t1[2]).key == ParseBlobLineage(t2[0]).key);
}

/* When the name does not carry an identifiable timestep the lineage
   is UNRESOLVED, and an unresolved lineage must make the caller run the NN
   rather than reuse somebody else's cached answer. */
TEST_CASE("PredictionReuseLineageUnresolvableNamesAreRefused") {
  for (const char *name : {
           "",                       // nothing at all
           "blob",                   // one component, no timestep
           "a/b",                    // no component looks like a timestep
           "E_x/chunk_0",            // the KEY itself, with no step
           "step00000/step00001/x",  // ambiguous: two timestep components
       }) {
    const BlobLineage l = ParseBlobLineage(name);
    REQUIRE_FALSE(l.resolved);
    REQUIRE(l.key.empty());
    REQUIRE(l.timestep < 0);
  }
}

/* `chunk_<i>` ends in digits like a timestep does. Mistaking it for one
   would collapse every chunk of a field onto one lineage. */
TEST_CASE("PredictionReuseLineageChunkComponentIsNotATimestep") {
  const BlobLineage a = ParseBlobLineage("step00000/E_x/chunk_0");
  const BlobLineage b = ParseBlobLineage("step00000/E_x/chunk_7");
  REQUIRE(a.resolved);
  REQUIRE(b.resolved);
  REQUIRE(a.timestep == 0);
  REQUIRE(b.timestep == 0);
  REQUIRE(a.key != b.key);
}

/* Leading zeros are decimal, not octal: step_00025 is 25, and step00010 is
   10. strtoll with base 8 would make them 21 and 8. */
TEST_CASE("PredictionReuseLineageLeadingZerosAreDecimal") {
  REQUIRE(ParseBlobLineage("cbx/step_00025/chunk_0").timestep == 25);
  REQUIRE(ParseBlobLineage("step00010/E_x/chunk_0").timestep == 10);
  REQUIRE(ParseBlobLineage("plt00009/d/chunk_0").timestep == 9);
}
