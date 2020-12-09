from testflows.core import *
from testflows.asserts import error

from rbac.requirements import *
from rbac.helper.common import *
import rbac.helper.errors as errors

@TestSuite
def privileges_granted_directly(self, node=None):
    """Check that a user is able to execute `DROP ROLE` with privileges are granted directly.
    """

    user_name = f"user_{getuid()}"

    if node is None:
        node = self.context.node

    with user(node, f"{user_name}"):

        Suite(run=drop_role, flags=TE,
            examples=Examples("privilege grant_target_name user_name", [
                tuple(list(row)+[user_name,user_name]) for row in drop_role.examples
            ], args=Args(name="privilege={privilege}", format_name=True)))

@TestSuite
def privileges_granted_via_role(self, node=None):
    """Check that a user is able to execute `DROP ROLE` with privileges are granted through a role.
    """

    user_name = f"user_{getuid()}"
    role_name = f"role_{getuid()}"

    if node is None:
        node = self.context.node

    with user(node, f"{user_name}"), role(node, f"{role_name}"):

        with When("I grant the role to the user"):
            node.query(f"GRANT {role_name} TO {user_name}")

        Suite(run=drop_role, flags=TE,
            examples=Examples("privilege grant_target_name user_name", [
                tuple(list(row)+[role_name,user_name]) for row in drop_role.examples
            ], args=Args(name="privilege={privilege}", format_name=True)))

@TestOutline(Suite)
@Examples("privilege",[
    ("ACCESS MANAGEMENT",),
    ("DROP ROLE",),
])
def drop_role(self, privilege, grant_target_name, user_name, node=None):
    """Check that user is only able to execute `DROP ROLE` when they have the necessary privilege.
    """
    exitcode, message = errors.not_enough_privileges(name=user_name)

    if node is None:
        node = self.context.node

    with Scenario("DROP ROLE without privilege"):
        drop_role_name = f"drop_role_{getuid()}"
        with role(node, drop_role_name):

            with When("I check the user can't drop a role"):
                node.query(f"DROP ROLE {drop_role_name}", settings=[("user",user_name)],
                    exitcode=exitcode, message=message)

    with Scenario("DROP ROLE with privilege"):
        drop_role_name = f"drop_role_{getuid()}"
        with role(node, drop_role_name):

            with When(f"I grant {privilege}"):
                node.query(f"GRANT {privilege} ON *.* TO {grant_target_name}")

            with Then("I check the user can drop a role"):
                node.query(f"DROP ROLE {drop_role_name}", settings = [("user", f"{user_name}")])

    with Scenario("DROP ROLE on cluster"):
        drop_role_name = f"drop_role_{getuid()}"

        try:
            with Given("I have a role on a cluster"):
                node.query(f"CREATE ROLE {drop_role_name} ON CLUSTER sharded_cluster")

            with When(f"I grant {privilege}"):
                node.query(f"GRANT {privilege} ON *.* TO {grant_target_name}")

            with Then("I check the user can drop a role"):
                node.query(f"DROP ROLE {drop_role_name} ON CLUSTER sharded_cluster", settings = [("user", f"{user_name}")])

        finally:
            with Finally("I drop the user"):
                node.query(f"DROP ROLE IF EXISTS {drop_role_name} ON CLUSTER sharded_cluster")

    with Scenario("DROP ROLE with revoked privilege"):
        drop_role_name = f"drop_role_{getuid()}"
        with role(node, drop_role_name):
            with When(f"I grant {privilege}"):
                node.query(f"GRANT {privilege} ON *.* TO {grant_target_name}")

            with And(f"I revoke {privilege}"):
                node.query(f"REVOKE {privilege} ON *.* FROM {grant_target_name}")

            with Then("I check the user can't drop a role"):
                node.query(f"DROP ROLE {drop_role_name}", settings=[("user",user_name)],
                    exitcode=exitcode, message=message)

@TestFeature
@Name("drop role")
@Requirements(
    RQ_SRS_006_RBAC_Privileges_DropRole("1.0"),
)
def feature(self, node="clickhouse1"):
    """Check the RBAC functionality of DROP ROLE.
    """
    self.context.node = self.context.cluster.node(node)

    Suite(run=privileges_granted_directly, setup=instrument_clickhouse_server_log)
    Suite(run=privileges_granted_via_role, setup=instrument_clickhouse_server_log)
